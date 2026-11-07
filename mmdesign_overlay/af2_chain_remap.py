#!/usr/bin/env python3
"""Remap AF2-Multimer binder-protocol collapsed PDBs to logical multi-chain IDs.

TNF trimer AF2 trajectories (observed):
  chain A has THREE contiguous segments of 152 CA each, separated by ~56-res gaps
    e.g. 1–152, 208–359, 415–566  → logical A,B,C
  chain B length ~131 = binder (logical D)

Never score interface as A_B on collapsed PDB.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def detect_layout(pdb_path: str) -> Dict:
    from collections import Counter
    ca = Counter()
    for line in Path(pdb_path).open():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca[line[21]] += 1
    return {"ca_counts": dict(ca), "n_chains": len(ca)}


def _ca_resseqs(pdb_path: str, chain: str) -> List[int]:
    out = []
    for line in Path(pdb_path).open():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
            out.append(int(line[22:26]))
    return out


def _segments(resseqs: List[int]) -> List[Tuple[int, int]]:
    """Return inclusive (start,end) contiguous segments in order of appearance."""
    if not resseqs:
        return []
    segs = []
    start = prev = resseqs[0]
    for r in resseqs[1:]:
        if r != prev + 1:
            segs.append((start, prev))
            start = r
        prev = r
    segs.append((start, prev))
    return segs


def tnfa_af2_segment_map(pdb_path: str) -> Dict[str, Tuple[int, int]]:
    """Map logical chain -> (resseq_lo, resseq_hi) on collapsed AF2 chain A."""
    segs = _segments(_ca_resseqs(pdb_path, "A"))
    if len(segs) != 3:
        raise ValueError(f"expected 3 TNF segments on AF2 chain A, got {segs}")
    for lo, hi in segs:
        if hi - lo + 1 != 152:
            raise ValueError(f"segment {lo}-{hi} length {hi-lo+1} != 152")
    return {"A": segs[0], "B": segs[1], "C": segs[2]}


def remap_tnfa_af2_pdb(in_pdb: str, out_pdb: str, binder_af2_chain: str = "B") -> Dict:
    layout = detect_layout(in_pdb)
    ca = layout["ca_counts"]
    if set(ca.keys()) >= {"A", "B", "C"} and ca.get("A", 0) <= 160:
        Path(out_pdb).write_text(Path(in_pdb).read_text())
        return {"status": "already_multichain", **layout, "out": out_pdb}

    if ca.get("A") != 456 or binder_af2_chain not in ca:
        raise ValueError(f"unexpected TNF AF2 layout {ca}; expected A=456 + binder")

    segmap = tnfa_af2_segment_map(in_pdb)
    # reverse lookup resseq -> (logical, local_1based)
    res_to_logical = {}
    for logical, (lo, hi) in segmap.items():
        for r in range(lo, hi + 1):
            res_to_logical[r] = (logical, r - lo + 1)

    out_lines = []
    atom_serial = 1
    for line in Path(in_pdb).open():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            if line.startswith("END") or line.startswith("TER"):
                continue
            out_lines.append(line)
            continue
        ch = line[21]
        resseq = int(line[22:26])
        if ch == "A":
            if resseq not in res_to_logical:
                # non-CA gap residue numbers should not appear; skip hydrogens etc. if any stray
                raise KeyError(f"A resseq {resseq} not in TNF segment map {segmap}")
            logical, local = res_to_logical[resseq]
            new = list(line)
            new[21] = logical
            new[22:26] = f"{local:4d}"
            new[6:11] = f"{atom_serial:5d}"
            out_lines.append("".join(new))
            atom_serial += 1
        elif ch == binder_af2_chain:
            new = list(line)
            new[21] = "D"
            new[6:11] = f"{atom_serial:5d}"
            out_lines.append("".join(new))
            atom_serial += 1
        else:
            raise KeyError(f"unexpected chain {ch}")
    out_lines.append("END\n")
    Path(out_pdb).write_text("".join(out_lines))
    return {"status": "remapped", **layout, "out": out_pdb, "segments": {k: list(v) for k, v in segmap.items()},
            "logical_chains": ["A", "B", "C", "D"]}


def binder_target_interface_string(binder_chain: str, target_chains: str) -> str:
    tgt = target_chains.replace(",", "")
    return f"{binder_chain}_{tgt}"


def contact_pairs_not_self(pdb_path: str, binder_chain: str, target_chains: List[str], cutoff: float = 5.0) -> Dict:
    import numpy as np
    coords = {c: [] for c in [binder_chain] + list(target_chains)}
    for line in Path(pdb_path).open():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        ch = line[21]
        if ch not in coords:
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        resseq = int(line[22:26])
        coords[ch].append((resseq, np.array(xyz)))
    b = coords[binder_chain]
    if not b:
        raise ValueError("no binder atoms")
    hits = []
    for tch in target_chains:
        for br, bc in b:
            for tr, tc in coords[tch]:
                if np.linalg.norm(bc - tc) <= cutoff:
                    hits.append((binder_chain, br, tch, tr))
    ag_ag = 0
    if len(target_chains) >= 2:
        t0, t1 = target_chains[0], target_chains[1]
        for _, c0 in coords[t0][::5]:
            for _, c1 in coords[t1][::5]:
                if np.linalg.norm(c0 - c1) <= cutoff:
                    ag_ag += 1
    return {
        "binder_target_contacts": len(hits),
        "sample_hits": hits[:10],
        "antigen_antigen_ca_hits_subsample": ag_ag,
        "ok": True,
    }
