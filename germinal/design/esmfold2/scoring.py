"""Binder-specific scores for an ESMFold2 complex.

Acceptance uses binder pLDDT and binder–target ipSAE aggregated as
mean_targets_of_mean_directions_v1 (mean of the two PAE directions on each
binder–target chain pair, then the equal-weight mean across target chains).

Global iPTM is recorded and is not an acceptance score. A trimer can post a
high global iPTM from target–target contacts while the binder interface is
empty. Missing, non-finite, or unmapped scores fail closed. There is no
positional fallback when chain ids are absent.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

AGGREGATION = "mean_targets_of_mean_directions_v1"


class ScoreError(ValueError):
    pass


def _finite_array(value, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def normalize_plddt(plddt: np.ndarray) -> np.ndarray:
    p = np.asarray(plddt, dtype=float).reshape(-1)
    if p.size == 0 or not np.isfinite(p).all():
        raise ScoreError("pLDDT missing or non-finite")
    if float(p.max()) > 1.5:
        p = p / 100.0
    if float(p.min()) < 0.0 or float(p.max()) > 1.0:
        raise ScoreError("pLDDT outside [0, 1] after normalization")
    return p


def directed_ipsae(pae: np.ndarray, src: np.ndarray, dst: np.ndarray, cutoff: float) -> dict:
    """Dunbrack ipSAE along one PAE direction. No CA-distance filter."""
    best = 0.0
    best_i = None
    best_n = 0
    for i in src:
        vals = pae[int(i), dst]
        keep = vals < cutoff
        n = int(keep.sum())
        if n == 0:
            continue
        if n >= 27:
            d0 = 1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8
        else:
            d0 = 1.0
        d0 = max(float(d0), 1.0)
        score = float(np.mean(1.0 / (1.0 + (vals[keep] / d0) ** 2)))
        if score > best:
            best = score
            best_i = int(i)
            best_n = n
    return {
        "ipsae": float(best),
        "aligned_residue_index": best_i,
        "n_residues_below_cutoff": int(best_n),
    }


def _index(chain_ids: Sequence[str], chain_id: str) -> np.ndarray:
    return np.array([i for i, c in enumerate(chain_ids) if c == chain_id], dtype=int)


def evaluate_complex(
    *,
    chain_ids: Sequence[str] | None,
    plddt,
    pae,
    expected: Sequence[tuple[str, str, int]],
    iptm_global: float | None = None,
    pae_cutoff: float = 10.0,
    min_binder_plddt: float | None = None,
    min_binder_ipsae: float | None = None,
) -> dict:
    """Score one fold.

    expected is (chain_id, role, length) with exactly one binder.
    """
    base = {
        "status": "FAIL_CLOSED",
        "aggregation": AGGREGATION,
        "global_iptm": None,
        "global_iptm_used_for_acceptance": False,
        "binder_plddt": None,
        "binder_target_ipsae": None,
        "ipsae_target_group_pooled_max": None,
        "pair_scores": None,
        "objective": None,
        "passes_thresholds": False,
        "chain_map": None,
    }
    if iptm_global is not None:
        try:
            g = float(iptm_global)
        except (TypeError, ValueError):
            g = float("nan")
        base["global_iptm"] = g if np.isfinite(g) else None

    if chain_ids is None:
        base["reason"] = "missing_chain_ids"
        return base
    ids = [str(c) for c in chain_ids]
    roles = {cid: role for cid, role, _length in expected}
    binders = [cid for cid, role, _length in expected if role == "binder"]
    targets = [cid for cid, role, _length in expected if role == "target"]
    if len(binders) != 1 or not targets:
        base["reason"] = "expected_one_binder_and_target"
        return base
    if len({cid for cid, _role, _length in expected}) != len(expected):
        base["reason"] = "duplicate_expected_chain_id"
        return base
    expected_n = sum(length for _cid, _role, length in expected)
    if len(ids) != expected_n:
        base["reason"] = "chain_id_length_mismatch"
        base["chain_map"] = {"n_tokens": len(ids), "expected_tokens": expected_n}
        return base
    counts: dict[str, int] = {}
    for cid in ids:
        counts[cid] = counts.get(cid, 0) + 1
    if set(counts) != set(roles):
        base["reason"] = "chain_id_set_mismatch"
        base["chain_map"] = {"observed": sorted(counts), "expected": sorted(roles)}
        return base
    for cid, _role, length in expected:
        if counts[cid] != int(length):
            base["reason"] = "chain_length_mismatch"
            base["chain_map"] = {
                "chain": cid,
                "observed": counts[cid],
                "expected": int(length),
            }
            return base

    try:
        p = normalize_plddt(plddt)
    except ScoreError as exc:
        base["reason"] = "invalid_plddt"
        base["detail"] = str(exc)
        return base
    if p.shape[0] != len(ids):
        base["reason"] = "plddt_length_mismatch"
        return base
    pae_arr = _finite_array(pae, "pae")
    if pae_arr is None or pae_arr.ndim != 2 or pae_arr.shape != (len(ids), len(ids)):
        base["reason"] = "missing_or_bad_pae"
        return base
    if float(pae_arr.min()) < 0.0:
        base["reason"] = "negative_pae"
        return base

    binder_id = binders[0]
    binder_idx = _index(ids, binder_id)
    binder_plddt = float(p[binder_idx].mean())
    pair_scores = []
    means = []
    for tid in targets:
        tidx = _index(ids, tid)
        forward = directed_ipsae(pae_arr, binder_idx, tidx, pae_cutoff)
        backward = directed_ipsae(pae_arr, tidx, binder_idx, pae_cutoff)
        mean_dir = float(0.5 * (forward["ipsae"] + backward["ipsae"]))
        means.append(mean_dir)
        pair_scores.append(
            {
                "binder": binder_id,
                "target": tid,
                "binder_to_target": forward,
                "target_to_binder": backward,
                "mean_directions": mean_dir,
                "max_directions": float(max(forward["ipsae"], backward["ipsae"])),
            }
        )
    target_idx = np.concatenate([_index(ids, tid) for tid in targets])
    pooled_bt = directed_ipsae(pae_arr, binder_idx, target_idx, pae_cutoff)
    pooled_tb = directed_ipsae(pae_arr, target_idx, binder_idx, pae_cutoff)
    pooled_max = float(max(pooled_bt["ipsae"], pooled_tb["ipsae"]))
    ipsae = float(np.mean(means))
    if not np.isfinite(binder_plddt) or not np.isfinite(ipsae) or not np.isfinite(pooled_max):
        base["reason"] = "non_finite_score"
        return base

    passes = True
    if min_binder_plddt is not None and binder_plddt < float(min_binder_plddt):
        passes = False
    if min_binder_ipsae is not None and ipsae < float(min_binder_ipsae):
        passes = False
    base.update(
        {
            "status": "SCORED",
            "reason": "ok",
            "binder_plddt": binder_plddt,
            "binder_target_ipsae": ipsae,
            "ipsae_target_group_pooled_max": pooled_max,
            "pair_scores": pair_scores,
            "objective": ipsae,
            "passes_thresholds": passes,
            "chain_map": {
                "binder": binder_id,
                "targets": list(targets),
                "lengths": {cid: counts[cid] for cid in counts},
            },
            "pae_cutoff": float(pae_cutoff),
        }
    )
    return base
