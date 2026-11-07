"""CPU tests for the ESMFold2 design backend. No weights and no GPU."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from germinal.design.esmfold2.campaign import (
    CampaignConfig,
    CampaignError,
    load_config,
    prepare_run,
    result_exit_code,
    run_campaign,
    write_result,
)
from germinal.design.esmfold2.constructs import (
    TNFA_DESIGN_152,
    TNFA_OFFICIAL,
    get_construct,
)
from germinal.design.esmfold2.framework import (
    FrameworkLockError,
    assert_framework_identical,
    cdr_positions_0,
    mutate_cdrs,
)
from germinal.design.esmfold2.msa import MsaPolicyError, prepare_chains
from germinal.design.esmfold2.policy import assert_process_has_no_af2, scan_forbidden
from germinal.design.esmfold2.runtime import runtime_status
from germinal.design.esmfold2.chainmap import read_result_chain_ids
from germinal.design.esmfold2.scoring import directed_ipsae, evaluate_complex
from germinal.utils.overlay_path import resolve_mmdesign_overlay
from run_germinal_esmfold2 import enforce_single_gpu, main


def _template(cdr_lengths=(11, 8, 18), fw_lengths=(25, 17, 38, 14), cdr_aa="S"):
    cdr = cdr_positions_0(cdr_lengths, fw_lengths)
    seq = ["G"] * (sum(cdr_lengths) + sum(fw_lengths))
    for i in cdr:
        seq[i] = cdr_aa
    return "".join(seq), cdr


def _a3m(directory: Path, name: str, sequence: str) -> Path:
    path = directory / name
    path.write_text(f">query\n{sequence}\n")
    return path


def _constant_fold(cross, iptm, plddt=0.9):
    def fold(chains, seed):
        ids = []
        for chain in chains:
            ids.extend([chain.id] * len(chain.sequence))
        n = len(ids)
        pae = np.full((n, n), 1.0)
        for i, left in enumerate(ids):
            for j, right in enumerate(ids):
                if left != right:
                    pae[i, j] = cross
        return {
            "chain_ids": ids,
            "plddt": np.full(n, plddt),
            "pae": pae,
            "iptm_global": iptm,
        }

    return fold


class FrameworkTests(unittest.TestCase):
    def test_positions_match_deployed_helper(self):
        path = ROOT / "mmdesign_overlay" / "framework_lock.py"
        spec = importlib.util.spec_from_file_location("deployed_framework_lock", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        got = cdr_positions_0()
        self.assertEqual(got, module.cdr_positions_0())
        self.assertEqual(got[:11], list(range(25, 36)))
        self.assertEqual(got[11:19], list(range(53, 61)))
        self.assertEqual(got[19:], list(range(99, 117)))

    def test_mutate_leaves_framework(self):
        template, cdr = _template()
        rng = np.random.default_rng(7)
        seq = template
        for _ in range(40):
            seq = mutate_cdrs(seq, template, cdr, rng, n_mut=3, omit="C")
            assert_framework_identical(seq, template, cdr)
            self.assertNotIn("C", seq)
            self.assertTrue(any(seq[i] != template[i] for i in cdr) or seq != template)

    def test_framework_mutation_raises(self):
        template, cdr = _template()
        broken = "A" + template[1:]
        with self.assertRaises(FrameworkLockError):
            assert_framework_identical(broken, template, cdr)


class MsaTests(unittest.TestCase):
    def test_trimer_reuses_query_and_binder_has_no_msa(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _a3m(Path(tmp), "tnf.a3m", TNFA_OFFICIAL)
            template, _cdr = _template()
            chains = prepare_chains(
                target_chains=[
                    ("A", TNFA_OFFICIAL, "tnfa_official"),
                    ("B", TNFA_OFFICIAL, "tnfa_official"),
                    ("C", TNFA_OFFICIAL, "tnfa_official"),
                ],
                binder_id="D",
                binder_sequence=template,
                msa_paths={"tnfa_official": str(path)},
            )
        targets = [c for c in chains if c.role == "target"]
        binder = next(c for c in chains if c.role == "binder")
        self.assertEqual(len({c.msa_path for c in targets}), 1)
        self.assertIsNone(binder.msa_path)
        self.assertTrue(all(c.msa_query_sha256 for c in targets))

    def test_refuses_to_slice_official_msa_onto_152(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _a3m(Path(tmp), "tnf.a3m", TNFA_OFFICIAL)
            with self.assertRaises(MsaPolicyError):
                prepare_chains(
                    target_chains=[("A", TNFA_DESIGN_152, "tnfa_official")],
                    binder_id="B",
                    binder_sequence=_template()[0],
                    msa_paths={"tnfa_official": str(path)},
                )

    def test_binder_msa_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _a3m(Path(tmp), "tnf.a3m", TNFA_OFFICIAL)
            with self.assertRaises(MsaPolicyError):
                prepare_chains(
                    target_chains=[("A", TNFA_OFFICIAL, "tnfa_official")],
                    binder_id="B",
                    binder_sequence=_template()[0],
                    msa_paths={"tnfa_official": str(path)},
                    binder_msa_path=str(path),
                )


class ScoringTests(unittest.TestCase):
    def test_directed_ipsae_matches_hand_value(self):
        pae = np.array([[0.0, 4.0], [4.0, 0.0]])
        score = directed_ipsae(pae, np.array([0]), np.array([1]), 10.0)
        self.assertAlmostEqual(score["ipsae"], 1.0 / 17.0)

    def test_target_contacts_do_not_create_binder_ipsae(self):
        ids = ["H"] + ["A"] * 8
        pae = np.full((9, 9), 0.5)
        pae[0, 1:] = 30.0
        pae[1:, 0] = 30.0
        scored = evaluate_complex(
            chain_ids=ids,
            plddt=np.full(9, 0.95),
            pae=pae,
            expected=(("A", "target", 8), ("H", "binder", 1)),
            iptm_global=0.9,
        )
        self.assertEqual(scored["status"], "SCORED")
        self.assertEqual(scored["binder_target_ipsae"], 0.0)
        self.assertFalse(scored["global_iptm_used_for_acceptance"])
        self.assertEqual(scored["global_iptm"], 0.9)
        self.assertEqual(scored["objective"], 0.0)

    def test_missing_chain_ids_and_nan_pae_fail_closed(self):
        expected = (("A", "target", 2), ("B", "binder", 1))
        missing = evaluate_complex(
            chain_ids=None,
            plddt=np.ones(3),
            pae=np.ones((3, 3)),
            expected=expected,
            iptm_global=0.8,
        )
        self.assertEqual(missing["status"], "FAIL_CLOSED")
        self.assertEqual(missing["reason"], "missing_chain_ids")
        self.assertIsNone(missing["objective"])
        nan = evaluate_complex(
            chain_ids=["A", "A", "B"],
            plddt=np.ones(3),
            pae=np.array([[0, 1, 1], [1, 0, np.nan], [1, 1, 0]]),
            expected=expected,
        )
        self.assertEqual(nan["status"], "FAIL_CLOSED")
        self.assertEqual(nan["reason"], "missing_or_bad_pae")

    def test_length_mismatch_fail_closed(self):
        scored = evaluate_complex(
            chain_ids=["A", "B"],
            plddt=np.ones(2),
            pae=np.ones((2, 2)),
            expected=(("A", "target", 3), ("B", "binder", 1)),
        )
        self.assertEqual(scored["status"], "FAIL_CLOSED")
        self.assertEqual(scored["reason"], "chain_id_length_mismatch")


    def test_monomer_asym_ids_map_through_chain_lookup(self):
        ids = [0] * 4 + [1] * 2
        plddt = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.85])
        pae = np.full((6, 6), 1.0)
        pae[:4, 4:] = 4.0
        pae[4:, :4] = 4.0
        scored = evaluate_complex(
            chain_ids=ids,
            chain_lookup={0: "A", 1: "B"},
            plddt=plddt,
            pae=pae,
            expected=(("A", "target", 4), ("B", "binder", 2)),
            iptm_global=0.99,
        )
        self.assertEqual(scored["status"], "SCORED")
        self.assertAlmostEqual(scored["binder_plddt"], 0.875)
        self.assertAlmostEqual(scored["binder_target_ipsae"], 1.0 / 17.0)
        self.assertEqual(scored["chain_map"]["chain_lookup"], {"0": "A", "1": "B"})
        self.assertFalse(scored["global_iptm_used_for_acceptance"])

    def test_oligomer_lookup_keeps_equal_length_targets_distinct(self):
        # Three tokens each for A and B, two for binder D. A and B share a length.
        ids = [0, 0, 0, 1, 1, 1, 2, 2]
        plddt = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8])
        pae = np.full((8, 8), 1.0)
        pae[6:, :3] = 4.0
        pae[:3, 6:] = 4.0
        pae[6:, 3:6] = 30.0
        pae[3:6, 6:] = 30.0
        scored = evaluate_complex(
            chain_ids=np.array(ids, dtype=np.int64),
            chain_lookup={np.int64(0): "A", np.int64(1): "B", np.int64(2): "D"},
            plddt=plddt,
            pae=pae,
            expected=(
                ("A", "target", 3),
                ("B", "target", 3),
                ("D", "binder", 2),
            ),
        )
        self.assertEqual(scored["status"], "SCORED")
        self.assertAlmostEqual(scored["binder_plddt"], 0.8)
        pairs = {row["target"]: row["mean_directions"] for row in scored["pair_scores"]}
        self.assertAlmostEqual(pairs["A"], 1.0 / 17.0)
        self.assertEqual(pairs["B"], 0.0)
        self.assertAlmostEqual(scored["binder_target_ipsae"], 0.5 / 17.0)

    def test_numeric_ids_without_lookup_fail_closed(self):
        scored = evaluate_complex(
            chain_ids=[0, 0, 1],
            plddt=np.ones(3),
            pae=np.ones((3, 3)),
            expected=(("A", "target", 2), ("B", "binder", 1)),
        )
        self.assertEqual(scored["status"], "FAIL_CLOSED")
        self.assertEqual(scored["reason"], "chain_id_set_mismatch")
        self.assertIsNone(scored["binder_plddt"])
        self.assertIsNone(scored["objective"])

    def test_ambiguous_or_incomplete_lookup_fail_closed(self):
        expected = (("A", "target", 2), ("B", "binder", 1))
        ambiguous = evaluate_complex(
            chain_ids=[0, 0, 1],
            chain_lookup={0: "A", 1: "A"},
            plddt=np.array([0.1, 0.1, 0.9]),
            pae=np.ones((3, 3)),
            expected=expected,
        )
        self.assertEqual(ambiguous["status"], "FAIL_CLOSED")
        self.assertEqual(ambiguous["reason"], "chain_lookup_ambiguous")
        self.assertIsNone(ambiguous["objective"])
        incomplete = evaluate_complex(
            chain_ids=[0, 0, 1],
            chain_lookup={0: "A"},
            plddt=np.ones(3),
            pae=np.ones((3, 3)),
            expected=expected,
        )
        self.assertEqual(incomplete["reason"], "chain_lookup_incomplete")
        self.assertEqual(incomplete["status"], "FAIL_CLOSED")

    def test_reader_uses_metadata_lookup_not_invented_letters(self):
        class Meta:
            chain_lookup = {0: "A", 1: "B"}

        class Complex:
            chain_id = np.array([0, 0, 1])
            metadata = Meta()

        ids, lookup = read_result_chain_ids(Complex())
        self.assertEqual(ids, ["0", "0", "1"])
        self.assertEqual(lookup, {"0": "A", "1": "B"})
        self.assertIsNone(read_result_chain_ids(None)[0])


class CampaignTests(unittest.TestCase):
    def _cfg(self, tmp: Path, **overrides) -> CampaignConfig:
        msa = _a3m(tmp, "tnf.a3m", TNFA_OFFICIAL)
        template, _cdr = _template()
        cfg = CampaignConfig(
            mode="design",
            construct="tnfa_official_monomer",
            msa_paths={"tnfa_official": str(msa)},
            binder_template=template,
            steps=3,
            mutations_per_step=1,
            seed=3,
            out_dir=str(tmp / "out"),
            min_binder_plddt=0.5,
            min_binder_ipsae=0.01,
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_search_prefers_binder_ipsae_over_global_iptm(self):
        template, cdr = _template()
        y_seq = list(template)
        for i in cdr:
            y_seq[i] = "Y"
        y_seq = "".join(y_seq)
        calls = []

        def fold(chains, seed):
            calls.append(seed)
            binder = next(c for c in chains if c.role == "binder")
            self.assertIsNone(binder.msa_path)
            self.assertTrue(all(c.msa_path for c in chains if c.role == "target"))
            good = binder.sequence == y_seq
            return _constant_fold(2.0 if good else 25.0, 0.05 if good else 0.99)(chains, seed)

        def propose(seq, _rng):
            if seq != y_seq:
                return y_seq
            return mutate_cdrs(seq, template, cdr, np.random.default_rng(0), 1, "C")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp), steps=2)
            result = run_campaign(cfg, fold, propose=propose)
        self.assertEqual(result["best_sequence"], y_seq)
        self.assertGreater(result["best"]["binder_target_ipsae"], 0.0)
        self.assertLess(result["best"]["global_iptm"], 0.5)
        self.assertFalse(result["global_iptm_used_for_acceptance"])
        self.assertFalse(result["af2_called"])
        self.assertFalse(result["differentiable_logits"])
        self.assertGreaterEqual(len(calls), 2)
        assert_process_has_no_af2()

    def test_trimer_validation_numeric_asym_ids(self):
        template, _cdr = _template()

        def fold(chains, seed):
            ids = []
            lookup = {}
            plddt = []
            for index, chain in enumerate(chains):
                ids.extend([index] * len(chain.sequence))
                lookup[index] = chain.id
                fill = 0.77 if chain.role == "binder" else 0.3
                plddt.extend([fill] * len(chain.sequence))
            n = len(ids)
            pae = np.full((n, n), 4.0)
            np.fill_diagonal(pae, 0.0)
            return {
                "chain_ids": ids,
                "chain_lookup": lookup,
                "plddt": np.array(plddt),
                "pae": pae,
                "iptm_global": 0.8,
            }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(
                Path(tmp),
                mode="validate",
                construct="tnfa_official_trimer",
                binder_sequence=template,
            )
            result = run_campaign(cfg, fold)
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(result["best"]["chain_map"]["targets"], ["A", "B", "C"])
        self.assertEqual(result["best"]["chain_map"]["binder"], "D")
        self.assertEqual(result["best"]["chain_map"]["lengths"], {"A": 157, "B": 157, "C": 157, "D": 131})
        self.assertAlmostEqual(result["best"]["binder_plddt"], 0.77)

    def test_campaign_scores_numeric_asym_ids(self):
        template, _cdr = _template()

        def fold(chains, seed):
            ids = []
            lookup = {}
            plddt = []
            for index, chain in enumerate(chains):
                ids.extend([index] * len(chain.sequence))
                lookup[index] = chain.id
                fill = 0.91 if chain.role == "binder" else 0.2
                plddt.extend([fill] * len(chain.sequence))
            n = len(ids)
            pae = np.full((n, n), 4.0)
            np.fill_diagonal(pae, 0.0)
            return {
                "chain_ids": ids,
                "chain_lookup": lookup,
                "plddt": np.array(plddt),
                "pae": pae,
                "iptm_global": 0.1,
            }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp), steps=0, mode="design")
            result = run_campaign(cfg, fold)
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertAlmostEqual(result["best"]["binder_plddt"], 0.91)
        self.assertGreater(result["best"]["binder_target_ipsae"], 0.0)
        self.assertEqual(result["best"]["chain_map"]["binder"], "B")

    def test_fail_closed_never_accepts(self):
        def fold(chains, seed):
            raw = _constant_fold(2.0, 0.9)(chains, seed)
            raw["pae"] = None
            return raw

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp), steps=2)
            result = run_campaign(cfg, fold)
            write_result(cfg.out_dir, result)
            self.assertEqual(result["status"], "NO_FINITE_SCORE")
            self.assertEqual(result_exit_code(result), 3)
            self.assertFalse((Path(cfg.out_dir) / "accepted.json").exists())
            self.assertFalse((Path(cfg.out_dir) / "best_finite.json").exists())

    def test_framework_break_stops_before_fold(self):
        template, _cdr = _template()
        calls = []

        def fold(chains, seed):
            calls.append(seed)
            return _constant_fold(2.0, 0.2)(chains, seed)

        def propose(seq, _rng):
            return "A" + seq[1:]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp), steps=2)
            with self.assertRaises(FrameworkLockError):
                run_campaign(cfg, fold, propose=propose)
            # The default path raises inside search after the initial fold.
            result_calls = len(calls)
        self.assertEqual(result_calls, 1)
        self.assertTrue(template.startswith("G"))

    def test_refuses_oligomer_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(Path(tmp), mode="design", construct="tnfa_official_trimer")
            with self.assertRaises(CampaignError):
                prepare_run(cfg)
            cfg.construct = "tfr1_official_dimer"
            with self.assertRaises(CampaignError):
                prepare_run(cfg)

    def test_trimer_validation_maps_three_chains(self):
        seen = {}

        def fold(chains, seed):
            seen["ids"] = [c.id for c in chains]
            seen["msa"] = [c.msa_path for c in chains]
            seen["lengths"] = [len(c.sequence) for c in chains]
            return _constant_fold(3.0, 0.7)(chains, seed)

        template, _cdr = _template()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(
                Path(tmp),
                mode="validate",
                construct="tnfa_official_trimer",
                binder_sequence=template,
                steps=99,
            )
            result = run_campaign(cfg, fold)
        self.assertEqual(seen["ids"], ["A", "B", "C", "D"])
        self.assertEqual(seen["lengths"], [157, 157, 157, 131])
        self.assertEqual(len(set(seen["msa"][:3])), 1)
        self.assertIsNone(seen["msa"][3])
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(get_construct("tnfa_official_trimer").design_allowed, False)

    def test_thresholds_unset_does_not_write_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(
                Path(tmp),
                steps=1,
                min_binder_plddt=None,
                min_binder_ipsae=None,
            )
            result = run_campaign(cfg, _constant_fold(3.0, 0.99))
            write_result(cfg.out_dir, result)
            self.assertEqual(result["status"], "SCORED_THRESHOLDS_UNSET")
            self.assertEqual(result_exit_code(result), 0)
            self.assertTrue((Path(cfg.out_dir) / "best_finite.json").exists())
            self.assertFalse((Path(cfg.out_dir) / "accepted.json").exists())


class EntrypointTests(unittest.TestCase):
    def test_source_and_worker_have_no_legacy_generator(self):
        hits = scan_forbidden(ROOT)
        self.assertEqual(hits, [])
        worker = (ROOT / "deploy" / "tencent" / "gpu_worker_esmfold2.sh").read_text()
        self.assertIn("run_germinal_esmfold2.py", worker)
        self.assertIn("flock -n", worker)
        self.assertIn('CUDA_VISIBLE_DEVICES="$GPU"', worker)
        subprocess.run(["bash", "-n", str(ROOT / "deploy/tencent/gpu_worker_esmfold2.sh")], check=True)

    def test_single_gpu_guard(self):
        old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            self.assertEqual(enforce_single_gpu("0"), "0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            with self.assertRaises(SystemExit):
                enforce_single_gpu("0")
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old

    def test_unknown_generator_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                yaml.safe_dump(
                    {"backend": "esmfold2", "af_params_dir": "/not/used", "mode": "design"}
                )
            )
            with self.assertRaises(CampaignError):
                load_config(path)

    def test_campaign_without_gpu_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            template, _cdr = _template()
            msa = _a3m(tmp_path, "tnf.a3m", TNFA_OFFICIAL)
            cfg_path = tmp_path / "cfg.yaml"
            out = tmp_path / "out"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "backend": "esmfold2",
                        "mode": "design",
                        "construct": "tnfa_official_monomer",
                        "msa_paths": {"tnfa_official": str(msa)},
                        "binder_template": template,
                        "steps": 1,
                        "out_dir": str(out),
                        "model_path": str(tmp_path / "missing-model"),
                        "ccd_cache": str(tmp_path / "missing-ccd"),
                    }
                )
            )
            rc = main(["--config", str(cfg_path), "--gpu", "0", "--out-dir", str(out)])
            self.assertEqual(rc, 2)
            blocked = json.loads((out / "blocked.json").read_text())
            self.assertIn("BLOCKED_NO_GPU", blocked["blockers"])
            self.assertFalse(blocked["af2_fallback"])
            self.assertFalse(blocked["started_design"])
            self.assertFalse((out / "accepted.json").exists())

    def test_this_machine_reports_no_gpu(self):
        status = runtime_status("/run/ti/models/esmfold2", "/run/ti/models/esmfold2")
        self.assertIn("BLOCKED_NO_GPU", status["blockers"])
        self.assertFalse(status["af2_fallback"])
        self.assertNotEqual(status["status"], "READY")

    def test_overlay_resolver_uses_vendored_snapshot(self):
        old = os.environ.pop("MMDESIGN_PROJECT_ROOT", None)
        try:
            path = resolve_mmdesign_overlay()
        finally:
            if old is not None:
                os.environ["MMDESIGN_PROJECT_ROOT"] = old
        self.assertEqual(path, ROOT / "mmdesign_overlay")
        self.assertTrue((path / "residue_map.py").is_file())

    def test_deployed_science_markers_still_present(self):
        self.assertIn(
            "SCIENCE_FIX_FW_HARDLOCK_STEP",
            (ROOT / "colabdesign/colabdesign/af/design.py").read_text(),
        )
        self.assertIn(
            "SCIENCE_FIX_SEQENT3",
            (ROOT / "germinal/design/design.py").read_text(),
        )
        self.assertIn(
            "SCIENCE_FIX_AF2_CHAIN_MAP",
            (ROOT / "germinal/utils/utils.py").read_text(),
        )
        self.assertIn(
            "fail-closed",
            (ROOT / "germinal/filters/filter_utils.py").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
