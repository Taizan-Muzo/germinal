"""Target constructs for ESMFold2 design and oligomer validation.

Sequences are the UniProt ranges used on the deployed campaign. The 152-aa
TNF design construct is official[5:] (drops N-terminal VRSSS) and is not a
silent trim of the 157-aa MSA. The incorrect 142-aa diagnostic truncation
is not a construct.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

TNFA_OFFICIAL = (
    "VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQ"
    "GCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAE"
    "INRPDYLDFAESGQVYFGIIAL"
)
TNFA_DESIGN_152 = TNFA_OFFICIAL[5:]
TFR1_OFFICIAL = (
    "CKGVEPKTECERLAGTESPVREEPGEDFPAARRLYWDDLKRKLSEKLDSTDFTGTIKLLNENSYVPRE"
    "AGSQKDENLALYVENQFREFKLSKVWRDQHFVKIQVKDSAQNSVIIVDKNGRLVYLVENPGGYVAYSK"
    "AATVTGKLVHANFGTKKDFEDLYTPVNGSIVIVRAGKITFAEKVANAESLNAIGVLIYMDQTKFPIVN"
    "AELSFFGHAHLGTGDPYTPGFPSFNHTQFPPSRSSGLPNIPVQTISRAAAEKLFGNMEGDCPSDWKTD"
    "STCRMVTSESKNVKLTVSNVLKEIKILNIFGVIKGFVEPDHYVVVGAQRDAWGPGAAKSGVGTALLLK"
    "LAQMFSDMVLKDGFQPSRSIIFASWSAGDFGSVGATEWLEGYLSSLHLKAFTYINLDKAVLGTSNFKV"
    "SASPLLYTLIEKTMQNVKHPVTGQFLYQDSNWASKVEKLTLDNAAFPFLAYSGIPAVSFCFCEDTDYP"
    "YLGTTMDTYKELIERIPELNKVARAAAEVAGQFVIKLTHDVELNLDYERYNSQLLSFVRDLNQYRADI"
    "KEMGLSLQWLYSARGDFFRATSRLTTDFGNAEKTDRFVMKKLNDRVMRVEYHFLSPYVSPKESPFRHV"
    "FWGSGSHTLPALLENLKLRKQNNGAFNETLFRNQLALATWTIQGAANALSGDVWDIDNEF"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


assert len(TNFA_OFFICIAL) == 157
assert _sha(TNFA_OFFICIAL) == "788eca607f0924e37a17c50cf53e6f7cd09164eeccf8a2f35e0fd6d904293950"
assert len(TNFA_DESIGN_152) == 152
assert TNFA_DESIGN_152 == TNFA_OFFICIAL[5:]
assert TNFA_DESIGN_152[143 - 6] == "D"
assert "RETPEGAEAK" in TNFA_DESIGN_152
assert len(TFR1_OFFICIAL) == 672
assert _sha(TFR1_OFFICIAL) == "5c886e4b99b913ccaa6f0a308db942448bb853da6bd340302350d6918d053d47"


@dataclass(frozen=True)
class ChainDef:
    id: str
    sequence: str
    msa_key: str


@dataclass(frozen=True)
class Construct:
    name: str
    design_allowed: bool
    chains: tuple[ChainDef, ...]
    binder_chain_id: str
    notes: str

    @property
    def target_lengths(self) -> dict[str, int]:
        return {c.id: len(c.sequence) for c in self.chains}


def _tnfa(seq: str, chain_id: str) -> ChainDef:
    return ChainDef(chain_id, seq, "tnfa_official" if seq == TNFA_OFFICIAL else "tnfa_design152")


CONSTRUCTS: dict[str, Construct] = {
    "tnfa_official_monomer": Construct(
        name="tnfa_official_monomer",
        design_allowed=True,
        chains=(_tnfa(TNFA_OFFICIAL, "A"),),
        binder_chain_id="B",
        notes=(
            "Design on UniProt P01375 Val77-Leu233 (157 aa). "
            "The MSA query must be this sequence."
        ),
    ),
    "tnfa_official_trimer": Construct(
        name="tnfa_official_trimer",
        design_allowed=False,
        chains=(
            _tnfa(TNFA_OFFICIAL, "A"),
            _tnfa(TNFA_OFFICIAL, "B"),
            _tnfa(TNFA_OFFICIAL, "C"),
        ),
        binder_chain_id="D",
        notes=(
            "Validation only. Three copies of the 157-aa chain share one MSA. "
            "Do not design on the trimer."
        ),
    ),
    "tnfa_design152_monomer": Construct(
        name="tnfa_design152_monomer",
        design_allowed=True,
        chains=(_tnfa(TNFA_DESIGN_152, "A"),),
        binder_chain_id="B",
        notes=(
            "PDB design construct, 152 aa = official[5:], Asp preserved at the "
            "UniProt 219 site. Requires an MSA whose query is exactly these "
            "152 residues. The 157-aa MSA is not sliced."
        ),
    ),
    "tfr1_official_monomer": Construct(
        name="tfr1_official_monomer",
        design_allowed=True,
        chains=(ChainDef("A", TFR1_OFFICIAL, "tfr1_official"),),
        binder_chain_id="B",
        notes=(
            "Design on UniProt P02786 Cys89-Phe760 (672 aa). "
            "Dimer validation is a separate mode."
        ),
    ),
    "tfr1_official_dimer": Construct(
        name="tfr1_official_dimer",
        design_allowed=False,
        chains=(
            ChainDef("A", TFR1_OFFICIAL, "tfr1_official"),
            ChainDef("B", TFR1_OFFICIAL, "tfr1_official"),
        ),
        binder_chain_id="C",
        notes="Validation only. Do not design on the TfR1 dimer.",
    ),
}


def get_construct(name: str) -> Construct:
    try:
        return CONSTRUCTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(CONSTRUCTS))
        raise KeyError(f"unknown construct {name!r}. Known: {known}") from exc
