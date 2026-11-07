"""Locate the deployed MMDesign overlay without touching that checkout.

The live Tencent tree keeps scientific helpers under the wrapper overlay.
This resolver uses that directory when it is present, and otherwise the
snapshot vendored in this fork. It does not reset or rewrite either tree.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEPLOYED = Path("/AIRvePFS/dair/sulixian-data/mmdesign_like_vhh/overlay")


def resolve_mmdesign_overlay() -> Path:
    """Return the overlay directory that contains residue_map.py.

    An explicit MMDESIGN_PROJECT_ROOT is authoritative and is not replaced
    with the vendored snapshot. When the variable is unset, the deployed
    path wins if it exists, so a machine that already has the dirty wrapper
    keeps using those files.
    """
    env = os.environ.get("MMDESIGN_PROJECT_ROOT")
    if env:
        path = Path(env) / "overlay"
        if not (path / "residue_map.py").is_file():
            raise FileNotFoundError(
                f"MMDESIGN_PROJECT_ROOT overlay is missing residue_map.py: {path}"
            )
        return path
    if (_DEPLOYED / "residue_map.py").is_file():
        return _DEPLOYED
    vendored = Path(__file__).resolve().parents[2] / "mmdesign_overlay"
    if (vendored / "residue_map.py").is_file():
        return vendored
    raise FileNotFoundError(
        "mmdesign overlay not found. Set MMDESIGN_PROJECT_ROOT to the wrapper "
        "that contains overlay/, or keep mmdesign_overlay/ in this repo."
    )
