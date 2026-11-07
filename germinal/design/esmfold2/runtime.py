"""Runtime readiness for an ESMFold2 campaign.

A missing GPU, weight directory, CCD directory, or ESMFold2 package is a
hard block. This module never substitutes another structure model.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def torch_importable() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def esmfold2_importable() -> bool:
    try:
        from esm.models.esmfold2 import (  # noqa: F401
            ESMFold2InputBuilder,
            EsmFold2Model,
            ProteinInput,
            StructurePredictionInput,
        )
    except Exception:
        return False
    return True


def gpu_visible() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    if not shutil.which("nvidia-smi"):
        return False
    proc = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and "GPU" in proc.stdout


def runtime_status(model_path: str, ccd_cache: str) -> dict:
    gpu = gpu_visible()
    model_ok = Path(model_path).is_dir()
    ccd_ok = Path(ccd_cache).is_dir()
    esm_ok = esmfold2_importable()
    torch_ok = torch_importable()
    blockers = []
    if not gpu:
        blockers.append("BLOCKED_NO_GPU")
    if not torch_ok:
        blockers.append("BLOCKED_NO_TORCH")
    if not esm_ok:
        blockers.append("BLOCKED_NO_ESMFOLD2_ENV")
    if not model_ok:
        blockers.append("BLOCKED_NO_MODEL")
    if not ccd_ok:
        blockers.append("BLOCKED_CCD_CACHE_NOT_DIRECTORY")
    return {
        "status": "READY" if not blockers else blockers[0],
        "blockers": blockers,
        "gpu_visible": gpu,
        "torch_installed": torch_ok,
        "esmfold2_importable": esm_ok,
        "model_path": str(model_path),
        "model_present": model_ok,
        "ccd_cache": str(ccd_cache),
        "ccd_cache_is_directory": ccd_ok,
        "af2_fallback": False,
        "structure_model": "esmfold2",
    }
