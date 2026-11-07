#!/usr/bin/env python3
"""ipSAE 1.0.1 adapter with explicit chain-pair selection, no text-file fallback.

The reported scalar is the mean of both directions for each requested binder /
target pair, then an equal-weight mean across requested target chains. This is
an explicit project aggregation, not KD or the score of a merged target chain.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import numpy as np

AGGREGATION = "mean_targets_of_mean_directions_v1"


def _finite_scalar(value, name):
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected one finite score, got {value!r}")
    v = float(a.reshape(-1)[0])
    if not 0 <= v <= 1:
        raise ValueError(f"{name}: score outside [0,1]: {v}")
    return v


def select_interface_scores(results, binder_chain, target_chains):
    """Pure parser for the installed package's metric -> chain1 -> chain2 API."""
    targets = list(target_chains)
    if not targets or len(set(targets)) != len(targets) or binder_chain in targets:
        raise ValueError("target chains must be nonempty, unique and disjoint from binder")
    stores = {
        "ipsae": results["ipsae_scores"]["ipsae_d0res_asym"],
        "pdockq2": results["pdockq_scores"]["pDockQ2"],
        "LIS": results["lis_scores"],
    }
    pairs = []
    for target in targets:
        pair = {"binder": binder_chain, "target": target, "directions": {}}
        for a, b in ((binder_chain, target), (target, binder_chain)):
            pair["directions"][f"{a}->{b}"] = {
                metric: _finite_scalar(store[a][b], f"{metric}:{a}->{b}")
                for metric, store in stores.items()
            }
        pair["mean_directions"] = {
            metric: float(np.mean([d[metric] for d in pair["directions"].values()]))
            for metric in stores
        }
        pairs.append(pair)
    return {
        **{metric: float(np.mean([p["mean_directions"][metric] for p in pairs])) for metric in stores},
        "binder_chain": binder_chain, "target_chains": targets,
        "aggregation": AGGREGATION, "pair_scores": pairs,
        "ipsae_metric": "ipsae_d0res_asym", "parser_version": 2,
    }


def write_af2_pae_json(path, pae, plddt=None):
    pae = np.asarray(pae, dtype=float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or not pae.size:
        raise ValueError(f"PAE must be a nonempty square matrix, got {pae.shape}")
    if not np.isfinite(pae).all() or (pae < 0).any():
        raise ValueError("PAE must be finite and nonnegative")
    if plddt is None:
        raise ValueError("per-residue pLDDT required; no scalar or PDB fallback")
    plddt = np.asarray(plddt, dtype=float).ravel()
    if len(plddt) != len(pae) or not np.isfinite(plddt).all():
        raise ValueError(f"invalid pLDDT vector for PAE length {len(pae)}")
    if (plddt < 0).any() or (plddt > 100).any():
        raise ValueError("pLDDT outside [0,100]")
    if float(plddt.max()) <= 1:
        plddt = plddt * 100
    Path(path).write_text(json.dumps({"predicted_aligned_error": pae.tolist(),
                                    "pae": pae.tolist(), "plddt": plddt.tolist()}, allow_nan=False))
    return str(path)


def compute_ipsae_pdockq2(pdb_path: str, pae: np.ndarray, plddt: Optional[np.ndarray] = None,
                         binder_chain: str = "B", target_chains: Sequence[str] = ("A",),
                         pae_cutoff: float = 10., dist_cutoff: float = 10.,
                         work_dir: Optional[str] = None) -> Dict[str, Any]:
    from ipsae import calculate_ipsae
    from residue_map import pdb_residue_table
    pdb = Path(pdb_path)
    residues = pdb_residue_table(pdb)
    expected = list(target_chains) + [binder_chain]
    if list(residues) != expected:
        raise ValueError(f"PDB chain/token order {list(residues)} != expected {expected}")
    n = sum(map(len, residues.values()))
    if np.asarray(pae).shape != (n, n):
        raise ValueError(f"PAE shape {np.asarray(pae).shape} != protein residues {(n,n)}")
    work = Path(work_dir or pdb.parent)
    work.mkdir(parents=True, exist_ok=True)
    pae_json = work / (pdb.stem + "_af2pae_for_ipsae.json")
    write_af2_pae_json(pae_json, pae, plddt)
    results = calculate_ipsae(str(pae_json), str(pdb), pae_cutoff, dist_cutoff)
    selected = select_interface_scores(results, binder_chain, target_chains)
    selected.update({"pae_json": str(pae_json), "pdb_sha256": hashlib.sha256(pdb.read_bytes()).hexdigest(),
                     "pae_sha256": hashlib.sha256(pae_json.read_bytes()).hexdigest(),
                     "pae_cutoff": pae_cutoff, "distance_cutoff": dist_cutoff,
                     "package": "ipsae", "note": "explicit_directed_pair_selection_no_file_fallback"})
    evidence_path = work / (pdb.stem + "_interface_scores.json")
    evidence_path.write_text(json.dumps(selected, indent=2, allow_nan=False))
    selected["evidence_path"] = str(evidence_path)
    return selected
