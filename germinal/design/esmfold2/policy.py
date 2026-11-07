"""Guards that keep the ESMFold2 generator from calling AF2."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Files that constitute the ESMFold2 production path.
SCAN_FILES = (
    "germinal/design/esmfold2",
    "run_germinal_esmfold2.py",
    "scripts/smoke_esmfold2_backend.py",
    "deploy/tencent/gpu_worker_esmfold2.sh",
    "configs/esmfold2",
)


def _join(*parts: str) -> str:
    return "".join(parts)


# Patterns are concatenated so this file does not contain the tokens it scans for.
FORBIDDEN = (
    re.compile(r"^\s*import\s+" + _join("colab", "design") + r"\b", re.M),
    re.compile(r"^\s*from\s+" + _join("colab", "design") + r"\b", re.M),
    re.compile(r"\b" + _join("mk_af", "design_model") + r"\b"),
    re.compile(_join("af2_", "multimer")),
    re.compile(r"\b" + _join("run_germinal", ".py") + r"\b"),
    re.compile(_join("germinal.design", ".design") + r"\b"),
    re.compile(r"^\s*import\s+" + _join("ja", "x") + r"\b", re.M),
    re.compile(r"^\s*from\s+" + _join("ja", "x") + r"\b", re.M),
    re.compile(_join("af_params", "_dir")),
)


def scan_forbidden(root: Path | None = None) -> list[str]:
    """Return human-readable hits. Empty means the ESMFold2 path is clean."""
    root = root or REPO_ROOT
    hits: list[str] = []
    files: list[Path] = []
    for rel in SCAN_FILES:
        path = root / rel
        if path.is_dir():
            for pattern in ("*.py", "*.yaml", "*.sh"):
                files.extend(sorted(path.rglob(pattern)))
        elif path.is_file():
            files.append(path)
        else:
            hits.append(f"missing scanned path: {rel}")
    for path in files:
        text = path.read_text()
        rel = path.relative_to(root)
        for pattern in FORBIDDEN:
            if pattern.search(text):
                hits.append(f"{rel}: matches {pattern.pattern}")
    return hits


def assert_source_has_no_af2(root: Path | None = None) -> None:
    hits = scan_forbidden(root)
    if hits:
        raise RuntimeError(
            "ESMFold2 path contains a forbidden AF2/ColabDesign reference:\n"
            + "\n".join(hits)
        )


def assert_process_has_no_af2() -> None:
    """Fail if this process has already imported the legacy generator stack."""
    legacy = _join("colab", "design")
    array_lib = _join("ja", "x")
    weight_tag = _join("af2_", "multimer")
    fold_pkg = _join("alpha", "fold")
    bad = []
    for name in sys.modules:
        if name == legacy or name.startswith(legacy + "."):
            bad.append(name)
        elif name == array_lib or name.startswith(array_lib + "."):
            bad.append(name)
        elif weight_tag in name or name.startswith(fold_pkg):
            bad.append(name)
    if bad:
        raise RuntimeError(
            "ESMFold2 process imported forbidden modules: " + ", ".join(sorted(bad))
        )
