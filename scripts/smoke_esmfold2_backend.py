"""Smoke the ESMFold2 backend.

Runs the CPU unit tests, then reports READY or a hard block. A machine
without a GPU exits 2 with BLOCKED_NO_GPU. This script does not start a
design and does not call the ColabDesign entrypoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_germinal_esmfold2 import main


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/esmfold2_smoke"
    raise SystemExit(main(["--smoke", "--out-dir", out]))
