"""Validate Chai PAE/pLDDT token tensors against PDB residue counts.

Chai StructureCandidates:
  pae:    [num_tokens, num_tokens]
  plddt:  [num_tokens]   (already averaged atom→token inside chai_lab)

For protein-only complexes (no ligands/DNA), tokens normally equal polymer residues.
We NEVER invent a length-N pLDDT vector of ones or tile a scalar mean.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np


def pdb_ca_counts(pdb_path: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for line in Path(pdb_path).open():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        ch = line[21]
        counts[ch] = counts.get(ch, 0) + 1
    return counts


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def prepare_pae_plddt_for_ipsae(
    pae,
    plddt_per_token,
    pdb_path: str,
    binder_chain: str,
    target_chains: Sequence[str],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Return (pae, plddt_or_None, evidence). Raises if PAE unusable.

    pLDDT: use per-token vector only when len == PAE dim AND (PAE dim == total CA
    in selected chains OR we explicitly record token!=res pending and omit pLDDT).
    Scalar/mean pLDDT → omit (STAGE_PENDING), never broadcast/fabricate.
    """
    evidence: Dict[str, Any] = {
        "pdb": str(pdb_path),
        "binder_chain": binder_chain,
        "target_chains": list(target_chains),
    }
    pae_arr = to_numpy(pae)
    if pae_arr is None:
        raise RuntimeError("Chai pae_matrix missing")
    pae_arr = np.asarray(pae_arr, dtype=float)
    if pae_arr.ndim == 3:
        # accidental candidate dim
        pae_arr = pae_arr[0]
    if pae_arr.ndim != 2 or pae_arr.shape[0] != pae_arr.shape[1]:
        raise RuntimeError(f"PAE must be square 2D, got {pae_arr.shape}")
    T = int(pae_arr.shape[0])
    evidence["pae_shape"] = list(pae_arr.shape)

    ca = pdb_ca_counts(pdb_path)
    evidence["pdb_ca_counts"] = ca
    wanted = list(target_chains) + [binder_chain]
    n_res = sum(ca.get(c, 0) for c in wanted)
    evidence["n_res_selected_chains"] = n_res
    evidence["token_eq_res"] = (T == n_res)

    plddt_arr = to_numpy(plddt_per_token)
    if plddt_arr is not None:
        plddt_arr = np.asarray(plddt_arr, dtype=float).ravel()
    evidence["plddt_raw_size"] = None if plddt_arr is None else int(plddt_arr.size)

    if plddt_arr is None:
        evidence["plddt_status"] = "STAGE_PENDING_missing"
        return pae_arr, None, evidence
    if plddt_arr.size == 1:
        evidence["plddt_status"] = "STAGE_PENDING_scalar_mean_not_usable"
        evidence["note"] = "refusing to broadcast scalar mean to PAE length (would fabricate)"
        return pae_arr, None, evidence
    if plddt_arr.size != T:
        evidence["plddt_status"] = "STAGE_PENDING_len_mismatch"
        evidence["note"] = f"plddt {plddt_arr.size} != pae {T}; omit plddt, keep real PAE"
        return pae_arr, None, evidence
    if T != n_res:
        evidence["plddt_status"] = "STAGE_PENDING_token_ne_res"
        evidence["note"] = (
            f"tokens {T} != selected-chain CA {n_res}; omit plddt rather than guess mapping"
        )
        return pae_arr, None, evidence

    evidence["plddt_status"] = "ok_per_token_eq_res"
    return pae_arr, plddt_arr, evidence


def pick_binder_chain_ptm(per_chain_ptm, binder_chain_index: int = -1) -> Optional[float]:
    """Use real Chai per_chain_ptm; never invent 1.0."""
    arr = to_numpy(per_chain_ptm)
    if arr is None:
        return None
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        return None
    try:
        return float(arr[binder_chain_index])
    except Exception:
        return None
