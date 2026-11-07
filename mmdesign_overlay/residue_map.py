#!/usr/bin/env python3
"""Authoritative UniProt ↔ construct ↔ input-PDB ↔ model-seq ↔ output-PDB residue maps.

TNF construct choice (documented, not silent):
  Official identity: UniProt P01375 Val77–Leu233 (157 aa).
  Design construct: mature 1TNF coords residues 6–157 (152 aa) = official[5:157]
  (missing N-term VRSSS = UniProt 77–81). PDB residue R corresponds to official
  index R-1 and UniProt number R+76.
  CRITICAL: raw 1TNF has D→L at mature143 / UniProt219 (a configured hotspot).
  Production PDBs MUST restore Asp at that site with real coordinates (not rename-only).

TfR1 mono construct: 11QC chain A coords ~122–760 (missing UniProt 89–121).
  PDB residue R → official Cys89–Phe760 index = R-89; Chai 1-based = R-121.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib
import json

AA3 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

# Official sequences (immutable)
TNFA_OFFICIAL = (
    "VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQ"
    "GCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAE"
    "INRPDYLDFAESGQVYFGIIAL"
)
assert len(TNFA_OFFICIAL) == 157
assert hashlib.sha256(TNFA_OFFICIAL.encode()).hexdigest() == (
    "788eca607f0924e37a17c50cf53e6f7cd09164eeccf8a2f35e0fd6d904293950"
)

# Design construct subsequence (mature 6–157)
TNFA_CONSTRUCT = TNFA_OFFICIAL[5:]  # 152 aa; starts at UniProt82 / PDB6
assert len(TNFA_CONSTRUCT) == 152
assert TNFA_CONSTRUCT[143 - 6] == "D"  # UniProt219 / PDB143 must be D in construct seq


@dataclass(frozen=True)
class ResidueRef:
    uniprot_pos: int          # 1-based UniProt
    official_idx0: int        # 0-based in official construct FASTA
    construct_idx0: Optional[int]  # 0-based in design construct seq (None if missing coords)
    pdb_resseq: Optional[int] # residue number in input PDB construct
    chai_1based: Optional[int]# 1-based index after Chai sequence renumber
    letter: str
    in_coords: bool
    note: str = ""


def tnfa_map() -> List[ResidueRef]:
    """Full official Val77–Leu233 map with construct/PDB/Chai fields."""
    out = []
    for i, aa in enumerate(TNFA_OFFICIAL):
        uniprot = 77 + i
        # construct starts at official index 5 (UniProt 82 / PDB 6)
        if i < 5:
            out.append(ResidueRef(uniprot, i, None, None, None, aa, False,
                                  "missing_Nterm_VRSSS_in_1TNF_coords"))
            continue
        construct_idx0 = i - 5
        pdb_resseq = 6 + construct_idx0  # PDB numbering for mature 1TNF
        chai_1based = construct_idx0 + 1
        note = ""
        if uniprot == 219:
            note = "hotspot_site_must_be_ASP_not_LEU_from_raw_1TNF"
        out.append(ResidueRef(uniprot, i, construct_idx0, pdb_resseq, chai_1based, aa, True, note))
    return out


def tnfa_pdb_to_chai(pdb_resseq: int) -> int:
    """Input-PDB residue number → Chai model-sequence 1-based."""
    if pdb_resseq < 6 or pdb_resseq > 157:
        raise ValueError(f"TNF PDB resseq {pdb_resseq} outside construct 6–157")
    return pdb_resseq - 5


def tnfa_hotspots_input_pdb() -> List[str]:
    """Configured Germinal/AF2 hotspot strings on input PDB numbering (chain A)."""
    return ["A31", "A32", "A87", "A115", "A143"]


def tnfa_hotspots_chai_1based() -> List[Tuple[str, int, str]]:
    """(letter, chai_1based, input_tag) for Chai restraints / contact checks."""
    m = {r.pdb_resseq: r for r in tnfa_map() if r.in_coords}
    out = []
    for tag in tnfa_hotspots_input_pdb():
        res = int(tag[1:])
        r = m[res]
        out.append((r.letter, r.chai_1based, tag))
    return out


def tfr1_mono_pdb_to_chai(pdb_resseq: int, construct: str = "mono") -> int:
    """11QC-style residue number → Chai 1-based.
    mono (122–760): chai = pdb - 121
    trunc (168–651): chai = pdb - 167
    """
    if construct == "mono" and 122 <= pdb_resseq <= 760:
        return pdb_resseq - 121
    if construct == "trunc" and 168 <= pdb_resseq <= 651:
        return pdb_resseq - 167
    raise ValueError(construct)


def tfr1_hotspots_chai(construct: str = "mono") -> List[Tuple[str, int, str]]:
    # Geometric Tf-facing set on chain A (input PDB numbering)
    tags = ["A208", "A215", "A348", "A350", "A611"]
    # letters from official P02786 (verified separately in tests against FASTA)
    letters = {208: "R", 215: "N", 348: "N", 350: "E", 611: "Y"}  # verified P02786 / input 11QC
    out = []
    for tag in tags:
        res = int(tag[1:])
        out.append((letters.get(res, "X"), tfr1_mono_pdb_to_chai(res, construct), tag))
    return out


# AF2-Multimer binder protocol collapses all target chains into chain A.
# Observed AF2 PDB ranges include chain-break gaps; these are NOT flat token indices.
# For other layouts use tnfa_af2_segment_map(actual_pdb), never assume these offsets.
TNFA_AF2_CHAIN_RANGES = {
    "A": (1, 152),
    "B": (208, 359),
    "C": (415, 566),
}
TNFA_AF2_BINDER_CHAIN = "B"
TNFA_LOGICAL_BINDER = "D"


def af2_collapsed_to_logical_chain(resseq: int, af2_chain: str, target: str = "tnfa") -> Tuple[str, int]:
    """Map AF2 trajectory chain/resseq → logical chain ID + 1-based construct resseq."""
    if target != "tnfa":
        raise NotImplementedError(target)
    if af2_chain == TNFA_AF2_BINDER_CHAIN:
        return TNFA_LOGICAL_BINDER, resseq
    if af2_chain != "A":
        raise KeyError(f"unexpected AF2 chain {af2_chain}")
    for logical, (lo, hi) in TNFA_AF2_CHAIN_RANGES.items():
        if lo <= resseq <= hi:
            return logical, resseq - lo + 1  # back to 1..152 construct numbering for Chai-like
    raise KeyError(f"AF2 A resseq {resseq} outside TNF trimer ranges")


def dump_tnfa_table(path: str) -> None:
    rows = [asdict(r) for r in tnfa_map()]
    with open(path, "w") as f:
        json.dump({
            "official_uniprot": "P01375",
            "official_range": "Val77-Leu233",
            "official_len": 157,
            "construct_choice": "mature_1TNF_6_157_with_D219_restored",
            "construct_len": 152,
            "construct_equals_official_slice": "official[5:157]",
            "missing_nterm": "VRSSS (UniProt77-81)",
            "d219_policy": "must_rebuild_ASP_sidechain_not_rename_only",
            "chai_hotspots_1based": [{"letter": a, "pos": p, "from": t} for a, p, t in tnfa_hotspots_chai_1based()],
            "af2_collapsed_ranges": TNFA_AF2_CHAIN_RANGES,
            "residues": rows,
        }, f, indent=2)


if __name__ == "__main__":
    dump_tnfa_table("/tmp/tnfa_residue_map.json")
    print("chai hotspots", tnfa_hotspots_chai_1based())


def pdb_residue_table(path):
    """Ordered protein CA residues by actual PDB chain and (resseq, insertion code).

    Used at model boundaries, so unsupported residues/models fail explicitly.
    No BioPython, model imports, scoring, or file writes.
    """
    from pathlib import Path
    chains = {}
    seen = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        if line[16] not in (" ", "A"):
            continue
        chain, key = line[21], (int(line[22:26]), line[26].strip())
        aa = AA3.get(line[17:20].strip())
        if aa is None:
            raise ValueError(f"unsupported protein residue: {line[:27]}")
        if (chain, key) in seen:
            if seen[chain, key] != aa:
                raise ValueError(f"ambiguous residue {chain}{key}")
            continue
        seen[chain, key] = aa
        chains.setdefault(chain, []).append({"resseq": key[0], "icode": key[1], "letter": aa})
    if not chains:
        raise ValueError(f"no protein CA residues: {path}")
    return chains


def map_hotspots_to_output(input_pdb, output_pdb, target_chains, hotspots):
    """Verify chain sequences and map input PDB IDs to output chain positions.

    Rosetta contact helpers consume within-chain positions, not raw PDB numbers.
    No alignment guessing: a deletion, mutation, unknown chain or insertion code
    must be handled explicitly before this boundary can pass.
    """
    import re
    source, output = pdb_residue_table(input_pdb), pdb_residue_table(output_pdb)
    targets = target_chains.split(",") if isinstance(target_chains, str) else list(target_chains)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target chains must be unique and nonempty")
    for ch in targets:
        a, b = source[ch], output[ch]
        if ''.join(r['letter'] for r in a) != ''.join(r['letter'] for r in b):
            raise ValueError(f"input/output target sequence mismatch: chain {ch}")
    mapped, evidence = {}, []
    for tag in str(hotspots or "").split(","):
        if not tag.strip():
            continue
        match = re.fullmatch(r"([A-Za-z0-9])(\d+)(?:-(\d+))?", tag.strip())
        if not match:
            raise ValueError(f"unsupported hotspot syntax {tag!r}")
        ch, first, last = match.groups()
        if ch not in targets:
            raise ValueError(f"hotspot chain {ch} not in target chains {targets}")
        first, last = int(first), int(last or first)
        if last < first:
            raise ValueError(f"reversed hotspot range {tag}")
        for number in range(first, last + 1):
            found = [(i, r) for i, r in enumerate(source[ch]) if r['resseq'] == number]
            if len(found) != 1 or found[0][1]['icode']:
                raise ValueError(f"missing/ambiguous input hotspot {ch}{number}")
            i, residue = found[0]
            mapped.setdefault(ch, []).append(i + 1)
            evidence.append({"chain": ch, "input_resseq": number, "letter": residue['letter'],
                             "output_resseq": output[ch][i]['resseq'], "output_position": i + 1})
    return {"positions": {ch: sorted(set(pos)) for ch, pos in mapped.items()},
            "residues": evidence, "version": 2}
