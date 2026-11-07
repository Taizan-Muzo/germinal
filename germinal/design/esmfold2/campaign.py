"""Campaign driver: monomer design or oligomer validation, ESMFold2 only."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from germinal.design.esmfold2.constructs import Construct, get_construct
from germinal.design.esmfold2.framework import (
    FrameworkLockError,
    assert_framework_identical,
    binder_length,
    cdr_positions_0,
    randomize_cdrs,
)
from germinal.design.esmfold2.msa import MsaPolicyError, prepare_chains
from germinal.design.esmfold2.search import run_search


class CampaignError(ValueError):
    pass


@dataclass
class CampaignConfig:
    backend: str = "esmfold2"
    mode: str = "design"
    construct: str = "tnfa_official_monomer"
    msa_paths: dict = field(default_factory=dict)
    binder_template: str = ""
    binder_sequence: str | None = None
    fw_lengths: list = field(default_factory=lambda: [25, 17, 38, 14])
    cdr_lengths: list = field(default_factory=lambda: [11, 8, 18])
    omit_aas: str = "C"
    require_target_msa: bool = True
    binder_msa_path: str | None = None
    model_path: str = "/run/ti/models/esmfold2"
    ccd_cache: str = "/run/ti/models/esmfold2"
    num_loops: int = 20
    num_sampling_steps: int = 200
    msa_max_depth: int = 1024
    steps: int = 8
    mutations_per_step: int = 2
    temperature_start: float = 0.02
    temperature_end: float = 0.005
    randomize_cdrs: bool = False
    seed: int = 1
    out_dir: str = "outputs/esmfold2"
    min_binder_plddt: float | None = None
    min_binder_ipsae: float | None = None
    pae_cutoff: float = 10.0
    hotspots_metadata: str = ""


def load_config(path: str | Path) -> CampaignConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise CampaignError("config must be a mapping")
    if raw.get("backend") != "esmfold2":
        raise CampaignError("backend must be esmfold2")
    unknown = set(raw) - set(CampaignConfig.__dataclass_fields__)
    if unknown:
        raise CampaignError(f"unknown config keys: {sorted(unknown)}")
    return CampaignConfig(**raw)


def _thresholds_configured(cfg: CampaignConfig) -> bool:
    return cfg.min_binder_plddt is not None or cfg.min_binder_ipsae is not None


def prepare_run(cfg: CampaignConfig):
    """Resolve sequences, MSA, and the framework lock. No model load."""
    if cfg.mode not in {"design", "validate"}:
        raise CampaignError("mode must be design or validate")
    construct: Construct = get_construct(cfg.construct)
    if cfg.mode == "design" and not construct.design_allowed:
        raise CampaignError(
            f"REFUSING design on {construct.name}. "
            "Design the monomer. Use mode=validate for the oligomer."
        )
    if cfg.mode == "validate" and cfg.randomize_cdrs:
        raise CampaignError("validation must not randomize CDRs")
    template = (cfg.binder_template or "").strip()
    if not template:
        raise CampaignError("binder_template is required")
    expected_len = binder_length(cfg.cdr_lengths, cfg.fw_lengths)
    if len(template) != expected_len:
        raise FrameworkLockError(
            f"binder_template length {len(template)} != "
            f"fw+cdr length {expected_len}"
        )
    cdr_idx = cdr_positions_0(cfg.cdr_lengths, cfg.fw_lengths)
    if cfg.mode == "validate":
        sequence = (cfg.binder_sequence or "").strip()
        if not sequence:
            raise CampaignError("validate mode requires binder_sequence")
    elif cfg.randomize_cdrs:
        if (cfg.binder_sequence or "").strip():
            raise CampaignError("set binder_sequence or randomize_cdrs, not both")
        rng = np.random.default_rng(cfg.seed)
        sequence = randomize_cdrs(template, cdr_idx, rng, cfg.omit_aas)
    else:
        sequence = (cfg.binder_sequence or template).strip()
    assert_framework_identical(sequence, template, cdr_idx)
    chains = prepare_chains(
        target_chains=[(c.id, c.sequence, c.msa_key) for c in construct.chains],
        binder_id=construct.binder_chain_id,
        binder_sequence=sequence,
        msa_paths={k: str(v) for k, v in cfg.msa_paths.items()},
        require_target_msa=cfg.require_target_msa,
        binder_msa_path=cfg.binder_msa_path,
    )
    expected = [(c.id, "target", len(c.sequence)) for c in construct.chains]
    expected.append((construct.binder_chain_id, "binder", len(sequence)))
    return {
        "construct": construct,
        "template": template,
        "cdr_idx0": cdr_idx,
        "start_sequence": sequence,
        "chains": chains,
        "expected": expected,
    }


def run_campaign(cfg: CampaignConfig, fold_fn, *, prepared=None, propose=None) -> dict:
    """Run design or a single validation fold.

    fold_fn(chains, seed) -> {chain_ids, plddt, pae, iptm_global}.
    Hotspot strings are copied into the record and are not forwarded.
    """
    prep = prepared or prepare_run(cfg)
    rng = np.random.default_rng(cfg.seed)
    if cfg.mode == "validate":
        record = run_search(
            parent_chains=prep["chains"],
            template=prep["template"],
            cdr_idx0=prep["cdr_idx0"],
            expected=prep["expected"],
            fold_fn=fold_fn,
            rng=rng,
            steps=0,
            mutations_per_step=1,
            omit_aas=cfg.omit_aas,
            temperature_start=cfg.temperature_start,
            temperature_end=cfg.temperature_end,
            base_seed=cfg.seed,
            pae_cutoff=cfg.pae_cutoff,
            min_binder_plddt=cfg.min_binder_plddt,
            min_binder_ipsae=cfg.min_binder_ipsae,
            propose=lambda seq, _rng: seq,
        )
    else:
        record = run_search(
            parent_chains=prep["chains"],
            template=prep["template"],
            cdr_idx0=prep["cdr_idx0"],
            expected=prep["expected"],
            fold_fn=fold_fn,
            rng=rng,
            steps=cfg.steps,
            mutations_per_step=cfg.mutations_per_step,
            omit_aas=cfg.omit_aas,
            temperature_start=cfg.temperature_start,
            temperature_end=cfg.temperature_end,
            base_seed=cfg.seed,
            pae_cutoff=cfg.pae_cutoff,
            min_binder_plddt=cfg.min_binder_plddt,
            min_binder_ipsae=cfg.min_binder_ipsae,
            propose=propose,
        )
    best = record["best"]
    configured = _thresholds_configured(cfg)
    if best is None:
        status = "NO_FINITE_SCORE"
    elif not configured:
        status = "SCORED_THRESHOLDS_UNSET"
    elif best.get("passes_thresholds"):
        status = "ACCEPTED"
    else:
        status = "SCORED_BELOW_THRESHOLD"
    result = {
        "status": status,
        "backend": "esmfold2",
        "mode": cfg.mode,
        "construct": cfg.construct,
        "construct_notes": prep["construct"].notes,
        "design_allowed": prep["construct"].design_allowed,
        "hotspots_metadata": cfg.hotspots_metadata,
        "hotspots_used_as_model_input": False,
        "coordinate_conditioning": False,
        "binder_msa": None,
        "target_msa": [
            {"id": c.id, "msa_path": c.msa_path, "query_sha256": c.msa_query_sha256}
            for c in prep["chains"]
            if c.role == "target"
        ],
        "num_loops": cfg.num_loops,
        "num_sampling_steps": cfg.num_sampling_steps,
        "protocol_note": (
            "num_loops/num_sampling_steps are explicit fold settings. "
            "20/200 matches the installed builder defaults used in the "
            "diagnostic audit, not a published contest protocol."
        ),
        "generator": record["generator"],
        "differentiable_logits": False,
        "af2_called": False,
        "global_iptm_used_for_acceptance": False,
        "aggregation": "mean_targets_of_mean_directions_v1",
        "thresholds_configured": configured,
        "best_sequence": record["best_sequence"],
        "best": best,
        "trajectory": record["trajectory"],
    }
    return result


def result_exit_code(result: dict) -> int:
    if result["status"] in {"ACCEPTED", "SCORED_THRESHOLDS_UNSET"}:
        return 0
    return 3


def write_result(out_dir: str | Path, result: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "campaign.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (out / "trajectory.jsonl").open("w") as handle:
        for row in result["trajectory"]:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    if result.get("best") is not None:
        (out / "best_finite.json").write_text(
            json.dumps(
                {
                    "sequence": result["best_sequence"],
                    "score": result["best"],
                    "status": result["status"],
                    "global_iptm_used_for_acceptance": False,
                },
                indent=2,
                allow_nan=False,
            )
        )
    accepted = out / "accepted.json"
    if result["status"] == "ACCEPTED":
        accepted.write_text(
            json.dumps(
                {
                    "sequence": result["best_sequence"],
                    "score": result["best"],
                    "aggregation": result["aggregation"],
                    "global_iptm_used_for_acceptance": False,
                },
                indent=2,
                allow_nan=False,
            )
        )
    elif accepted.exists():
        accepted.unlink()
