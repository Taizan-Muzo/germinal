"""Versioned per-design receipts. No model imports or scoring side effects."""
from __future__ import annotations
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

CONTRACT_VERSION = "mmdesign_fix_20261106_v2"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_result_receipt(root, location, data, structure_path):
    root, structure = Path(root), Path(structure_path)
    if data.get("audit_contract_version") != CONTRACT_VERSION:
        raise ValueError("cannot emit verified receipt for an unversioned execution")
    name = data["design_name"]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("invalid design name")
    sequence = data["trajectory_sequence"]
    config_path = root / "final_config.yaml"
    receipt = json_safe({
        "contract_version": CONTRACT_VERSION, "design_name": name,
        "experiment_name": root.name, "save_location": location,
        "structure_relative_path": str(structure.relative_to(root)),
        "structure_sha256": sha256_file(structure),
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "config_sha256": sha256_file(config_path),
        "scope": "production" if root.name.startswith("prod_") and not data.get("fixture") else "validation",
        "data": data,
    })
    text = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    event = root / "result_events" / name / f"{digest}.json"
    if not event.exists():
        atomic_text(event, text)
    atomic_text(root / "results" / f"{name}.json", text)
    return receipt
