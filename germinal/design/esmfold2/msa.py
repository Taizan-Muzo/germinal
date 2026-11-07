"""Target-only MSA wiring.

Target chains carry a real a3m whose query sequence equals that chain.
Identical target chains may share one file. The binder chain is refused
if a homologous MSA is configured. This module does not slice a longer
MSA down to a shorter construct.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class MsaPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedChain:
    id: str
    sequence: str
    role: str
    msa_path: str | None
    msa_query_sha256: str | None


def parse_a3m_query(path: Path | str) -> str:
    """First a3m sequence with insertions (lowercase) and gaps removed."""
    text = Path(path).read_text()
    started = False
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if started:
                break
            started = True
            continue
        if started:
            chunks.append(line.strip())
    if not started:
        raise MsaPolicyError(f"a3m has no header: {path}")
    raw = "".join(chunks)
    query = "".join(c for c in raw if c.isalpha() and c.isupper())
    if not query:
        raise MsaPolicyError(f"empty MSA query: {path}")
    return query


def _sha(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()


def prepare_chains(
    *,
    target_chains: list[tuple[str, str, str]],
    binder_id: str,
    binder_sequence: str,
    msa_paths: dict[str, str],
    require_target_msa: bool = True,
    binder_msa_path: str | None = None,
) -> list[PreparedChain]:
    """Build fold inputs.

    target_chains: (chain_id, sequence, msa_key) in complex order.
    msa_key selects msa_paths. The same key may be reused for identical
    sequences (TNF trimer, TfR1 dimer). A key whose query differs from the
    chain sequence is an error, including a 157-aa MSA against a 152-aa
    construct.
    """
    if binder_msa_path:
        raise MsaPolicyError(
            "binder homologous MSA is forbidden; ProteinInput.msa must be unset"
        )
    if not target_chains:
        raise MsaPolicyError("at least one target chain is required")
    ids = [binder_id] + [c[0] for c in target_chains]
    if len(ids) != len(set(ids)):
        raise MsaPolicyError(f"chain ids must be unique, got {ids}")
    if any(role_id == binder_id for role_id, _, _ in target_chains):
        raise MsaPolicyError("binder chain id collides with a target chain")

    prepared: list[PreparedChain] = []
    query_by_path: dict[str, str] = {}
    for chain_id, sequence, msa_key in target_chains:
        if not sequence or any(not c.isalpha() or not c.isupper() for c in sequence):
            raise MsaPolicyError(f"target {chain_id} sequence is not an uppercase protein sequence")
        path_value = msa_paths.get(msa_key) if msa_key else None
        if require_target_msa and not path_value:
            raise MsaPolicyError(
                f"target chain {chain_id} requires MSA key {msa_key!r}; "
                "refusing to fold a target with no MSA"
            )
        query_sha = None
        if path_value:
            path = str(Path(path_value))
            if path not in query_by_path:
                if not Path(path).is_file():
                    raise MsaPolicyError(f"MSA file missing for {chain_id}: {path}")
                query_by_path[path] = parse_a3m_query(path)
            query = query_by_path[path]
            if query != sequence:
                raise MsaPolicyError(
                    f"MSA query length {len(query)} != target {chain_id} length "
                    f"{len(sequence)}. Refusing to trim or pad the MSA."
                )
            query_sha = _sha(query)
        prepared.append(
            PreparedChain(
                id=chain_id,
                sequence=sequence,
                role="target",
                msa_path=str(Path(path_value)) if path_value else None,
                msa_query_sha256=query_sha,
            )
        )
    if not binder_sequence or any(not c.isalpha() or not c.isupper() for c in binder_sequence):
        raise MsaPolicyError("binder sequence is not an uppercase protein sequence")
    prepared.append(
        PreparedChain(
            id=binder_id,
            sequence=binder_sequence,
            role="binder",
            msa_path=None,
            msa_query_sha256=None,
        )
    )
    return prepared


def assert_binder_unpaired(chains: list[PreparedChain]) -> None:
    binders = [c for c in chains if c.role == "binder"]
    targets = [c for c in chains if c.role == "target"]
    if len(binders) != 1:
        raise MsaPolicyError(f"expected one binder chain, got {len(binders)}")
    if binders[0].msa_path is not None:
        raise MsaPolicyError("binder MSA path must stay unset")
    if not targets or any(c.msa_path is None for c in targets):
        raise MsaPolicyError("every target chain must carry an MSA path")
