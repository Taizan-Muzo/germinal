#!/usr/bin/env python3
"""Framework hard-lock helpers for VHH CDR-only redesign.

CDR 0-based (fw=[25,17,38,14], cdr=[11,8,18]):
  25-35, 53-60, 99-116  → 1-based 26-36, 54-61, 100-117
"""
from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np

DEFAULT_CDR = [11, 8, 18]
DEFAULT_FW = [25, 17, 38, 14]


def cdr_positions_0(cdr_lengths=None, fw_lengths=None) -> List[int]:
    cdr_lengths = list(cdr_lengths or DEFAULT_CDR)
    fw_lengths = list(fw_lengths or DEFAULT_FW)
    cumulative = 0
    positions = []
    for i, cdr_length in enumerate(cdr_lengths):
        fw_len = fw_lengths[i] + cumulative
        positions.extend(range(fw_len, fw_len + cdr_length))
        cumulative = fw_len + cdr_length
    return positions


def fw_positions_0(binder_len: int = 131, cdr_lengths=None, fw_lengths=None) -> List[int]:
    cdr = set(cdr_positions_0(cdr_lengths, fw_lengths))
    return [i for i in range(binder_len) if i not in cdr]


def assert_framework_identical(seq: str, template: str, cdr_idx0: Sequence[int]) -> None:
    if len(seq) != len(template):
        raise AssertionError(f"length mismatch {len(seq)} vs {len(template)}")
    cdr = set(cdr_idx0)
    bad = [(i, seq[i], template[i]) for i in range(len(seq)) if i not in cdr and seq[i] != template[i]]
    if bad:
        raise AssertionError(f"framework mutations at {bad[:10]} (showing up to 10)")


def zero_fw_grad(grad: np.ndarray, cdr_idx0: Sequence[int], binder_axis: int = -2) -> Tuple[np.ndarray, dict]:
    """Zero framework gradients. grad shape (..., L, 20) or (B, L, 20).

    Returns (masked_grad, evidence_dict).
    """
    g = np.array(grad, copy=True)
    L = g.shape[binder_axis]
    fw = fw_positions_0(L, cdr_lengths=None)  # uses default; override via cdr set
    # rebuild fw from cdr_idx0 for exactness
    cdr = set(int(x) for x in cdr_idx0)
    fw = [i for i in range(L) if i not in cdr]
    # fancy index along binder axis
    indexer = [slice(None)] * g.ndim
    indexer[binder_axis] = fw
    fw_before = float(np.linalg.norm(g[tuple(indexer)]))
    g[tuple(indexer)] = 0.0
    cdr_indexer = [slice(None)] * g.ndim
    cdr_indexer[binder_axis] = sorted(cdr)
    cdr_norm = float(np.linalg.norm(g[tuple(cdr_indexer)]))
    evid = {
        "fw_grad_norm_before": fw_before,
        "fw_grad_norm_after": 0.0,
        "cdr_grad_norm": cdr_norm,
        "fw_count": len(fw),
        "cdr_count": len(cdr),
        "grad_finite": bool(np.isfinite(g).all()),
    }
    return g, evid


def hard_bias_framework(bias: np.ndarray, wt_aatype: np.ndarray, cdr_idx0: Sequence[int], lock_value: float = 1e7) -> np.ndarray:
    """Force framework positions to wild-type AA with huge positive bias; suppress others."""
    b = np.array(bias, copy=True)
    cdr = set(int(x) for x in cdr_idx0)
    for i, aa in enumerate(wt_aatype):
        if i in cdr:
            continue
        b[i, :] = -lock_value
        b[i, int(aa)] = lock_value
    return b


def lock_seq_params(seq_params: np.ndarray, wt_onehot: np.ndarray, cdr_idx0: Sequence[int]) -> np.ndarray:
    """Overwrite framework logits/one-hot with template; leave CDR free."""
    x = np.array(seq_params, copy=True)
    cdr = set(int(x) for x in cdr_idx0)
    # x shape (1, L, 20) or (L, 20)
    if x.ndim == 3:
        L = x.shape[1]
        for i in range(L):
            if i not in cdr:
                x[:, i, :] = wt_onehot[i]
    elif x.ndim == 2:
        for i in range(x.shape[0]):
            if i not in cdr:
                x[i, :] = wt_onehot[i]
    else:
        raise ValueError(f"unexpected seq param ndim {x.ndim}")
    return x
