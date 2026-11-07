"""Map ESMFold2 token labels onto the campaign chain ids.

Observed convention, ESM 3.4.2 ``esm.models.esmfold2.output``
(``build_molecular_complex_from_features``):

- ``MolecularComplex.chain_id`` is an int64 per token. The value is
  ``ChainInfo.asym_id``, assigned 0, 1, 2, ... in ``ProteinInput`` order.
  It is not the string passed to ``ProteinInput(id=...)``.
- That string is stored separately as
  ``MolecularComplex.metadata.chain_lookup[asym_id]``.

Scoring uses the lookup. It does not rename ``0`` to ``A`` by position,
and it does not fall back to concatenating chains when the lookup is
missing or two asym ids share one name.
"""
from __future__ import annotations

from typing import Sequence

CONVENTION = "esmfold2_asym_id_via_chain_lookup"


def label(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    return str(value).strip()


def read_result_chain_ids(complex_obj) -> tuple[list[str] | None, dict[str, str] | None]:
    """Read raw token labels and the asym_id → input-id lookup, if present."""
    if complex_obj is None or not hasattr(complex_obj, "chain_id"):
        return None, None
    raw = complex_obj.chain_id
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().tolist()
    else:
        raw = list(raw)
    ids = [label(item) for item in raw]
    meta = getattr(complex_obj, "metadata", None)
    lookup_raw = getattr(meta, "chain_lookup", None) if meta is not None else None
    if not lookup_raw:
        return ids, None
    lookup = {label(key): label(val) for key, val in dict(lookup_raw).items()}
    return ids, lookup


def resolve_model_chain_ids(
    chain_ids: Sequence | None,
    chain_lookup: dict | None,
    expected_ids: set[str],
) -> tuple[list[str] | None, dict]:
    """Translate model labels to campaign ids.

    Returns ``(mapped_ids, evidence)``. ``mapped_ids`` is None when the
    map cannot be applied. Callers must fail closed in that case.
    """
    if chain_ids is None:
        return None, {"reason": "missing_chain_ids", "convention": CONVENTION}
    raw = [label(item) for item in chain_ids]
    evidence = {
        "convention": CONVENTION,
        "model_chain_ids_unique": list(dict.fromkeys(raw)),
    }
    if chain_lookup:
        table = {label(key): label(val) for key, val in dict(chain_lookup).items()}
        evidence["chain_lookup"] = table
        if any(value == "" for value in table.values()):
            evidence["reason"] = "chain_lookup_blank"
            return None, evidence
        if len(table.values()) != len(set(table.values())):
            evidence["reason"] = "chain_lookup_ambiguous"
            return None, evidence
        mapped: list[str] = []
        for token in raw:
            if token not in table:
                evidence["reason"] = "chain_lookup_incomplete"
                evidence["unmapped_model_chain_id"] = token
                return None, evidence
            mapped.append(table[token])
        evidence["mapped_unique"] = list(dict.fromkeys(mapped))
        evidence["reason"] = "mapped"
        return mapped, evidence
    if set(raw) == set(expected_ids):
        evidence["reason"] = "campaign_ids_already"
        return raw, evidence
    evidence["reason"] = "chain_id_set_mismatch"
    evidence["observed"] = sorted(set(raw))
    evidence["expected"] = sorted(expected_ids)
    evidence["note"] = (
        "chain_id values are ESMFold2 asym ids. metadata.chain_lookup was "
        "absent, so those integers were not translated to campaign chain ids."
    )
    return None, evidence
