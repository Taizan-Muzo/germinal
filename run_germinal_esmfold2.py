"""ESMFold2 sequence-design entrypoint.

One GPU. Target MSA only. Framework-locked CDR search scored by ESMFold2
folds. Missing scores fail closed. This process does not load the
ColabDesign generator and does not substitute another folding model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from germinal.design.esmfold2.campaign import (
    CampaignError,
    load_config,
    prepare_run,
    result_exit_code,
    run_campaign,
    write_result,
)
from germinal.design.esmfold2.framework import FrameworkLockError
from germinal.design.esmfold2.msa import MsaPolicyError
from germinal.design.esmfold2.policy import (
    assert_process_has_no_af2,
    assert_source_has_no_af2,
)
from germinal.design.esmfold2.runtime import runtime_status


DEFAULT_MODEL = "/run/ti/models/esmfold2"


def enforce_single_gpu(gpu: str) -> str:
    text = str(gpu).strip()
    if not text.isdigit() or "," in text or " " in text:
        raise SystemExit(
            f"REFUSING gpu {gpu!r}. One worker per GPU; pass a single index."
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = text
    else:
        parts = [part for part in visible.split(",") if part != ""]
        if parts != [text]:
            raise SystemExit(
                f"REFUSING CUDA_VISIBLE_DEVICES={visible!r} with --gpu {text}. "
                "One worker per GPU."
            )
    return text


def write_blocked(out_dir: Path, status: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(status)
    payload["af2_fallback"] = False
    payload["started_design"] = False
    (out_dir / "blocked.json").write_text(json.dumps(payload, indent=2))


def run_smoke(out_dir: Path) -> int:
    import subprocess

    assert_process_has_no_af2()
    assert_source_has_no_af2()
    repo = Path(__file__).resolve().parent
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_esmfold2_backend.py",
            "-v",
        ],
        cwd=repo,
        check=False,
    )
    status = runtime_status(DEFAULT_MODEL, DEFAULT_MODEL)
    status["unit_tests"] = "PASS" if proc.returncode == 0 else "FAIL"
    status["unit_test_rc"] = proc.returncode
    status["entrypoint"] = "run_germinal_esmfold2.py"
    status["one_worker_per_gpu"] = True
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SMOKE_STATUS.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if proc.returncode != 0:
        return 1
    if status["status"] != "READY":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ESMFold2 binder sequence design")
    parser.add_argument("--config", help="ESMFold2 campaign yaml")
    parser.add_argument("--gpu", help="Single physical GPU index")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate construct, framework lock, and target MSA. Do not fold.",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    assert_process_has_no_af2()
    assert_source_has_no_af2()

    if args.smoke:
        return run_smoke(Path(args.out_dir or "outputs/esmfold2_smoke"))
    if not args.config:
        parser.error("--config is required unless --smoke is set")

    try:
        cfg = load_config(args.config)
    except (CampaignError, TypeError, ValueError) as exc:
        print(f"CONFIG_ERROR {exc}", file=sys.stderr)
        return 4
    if args.out_dir:
        cfg.out_dir = args.out_dir

    if args.check_config:
        try:
            prep = prepare_run(cfg)
        except (CampaignError, FrameworkLockError, MsaPolicyError, KeyError) as exc:
            print(f"CONFIG_ERROR {exc}", file=sys.stderr)
            return 4
        summary = {
            "status": "CONFIG_OK",
            "construct": cfg.construct,
            "mode": cfg.mode,
            "binder_length": len(prep["start_sequence"]),
            "target_chains": [c.id for c in prep["chains"] if c.role == "target"],
            "binder_msa": None,
            "af2_fallback": False,
            "folded": False,
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.gpu is None:
        print("REFUSING implicit GPU. Pass --gpu <single index>.", file=sys.stderr)
        return 4
    try:
        enforce_single_gpu(args.gpu)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 4

    try:
        prepare_run(cfg)
    except (CampaignError, FrameworkLockError, MsaPolicyError, KeyError) as exc:
        print(f"CONFIG_ERROR {exc}", file=sys.stderr)
        return 4

    status = runtime_status(cfg.model_path, cfg.ccd_cache)
    if status["status"] != "READY":
        write_blocked(Path(cfg.out_dir), status)
        print(json.dumps(status, indent=2))
        return 2

    from germinal.design.esmfold2.fold import EsmFolder, load_esmfold2

    model, builder = load_esmfold2(cfg.model_path, cfg.ccd_cache, device="cuda")
    folder = EsmFolder(
        model,
        builder,
        num_loops=cfg.num_loops,
        num_sampling_steps=cfg.num_sampling_steps,
        msa_max_depth=cfg.msa_max_depth,
    )
    result = run_campaign(cfg, folder)
    write_result(cfg.out_dir, result)
    assert_process_has_no_af2()
    print(json.dumps({"status": result["status"], "out_dir": cfg.out_dir}))
    return result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
