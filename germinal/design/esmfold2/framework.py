"""CDR-only framework hard lock.

Positions follow the deployed VHH helper: framework lengths sit before each
CDR, and the final framework segment is whatever remains. Those residues
must stay identical to the template. This lock is a sequence assertion, not
an AF2 gradient mask.
"""
from __future__ import annotations

from typing import Sequence

AA20 = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_CDR = (11, 8, 18)
DEFAULT_FW = (25, 17, 38, 14)


class FrameworkLockError(ValueError):
    pass


def cdr_positions_0(
    cdr_lengths: Sequence[int] | None = None,
    fw_lengths: Sequence[int] | None = None,
) -> list[int]:
    cdr_lengths = list(cdr_lengths or DEFAULT_CDR)
    fw_lengths = list(fw_lengths or DEFAULT_FW)
    if len(fw_lengths) != len(cdr_lengths) + 1:
        raise FrameworkLockError(
            f"expected {len(cdr_lengths) + 1} framework segments for "
            f"{len(cdr_lengths)} CDRs, got {len(fw_lengths)}"
        )
    if any(int(x) < 0 for x in list(cdr_lengths) + list(fw_lengths)):
        raise FrameworkLockError("CDR and framework lengths must be non-negative")
    cumulative = 0
    positions: list[int] = []
    for i, cdr_length in enumerate(cdr_lengths):
        fw_len = int(fw_lengths[i]) + cumulative
        positions.extend(range(fw_len, fw_len + int(cdr_length)))
        cumulative = fw_len + int(cdr_length)
    return positions


def binder_length(cdr_lengths: Sequence[int], fw_lengths: Sequence[int]) -> int:
    cdr_positions_0(cdr_lengths, fw_lengths)
    return int(sum(cdr_lengths) + sum(fw_lengths))


def assert_framework_identical(
    seq: str,
    template: str,
    cdr_idx0: Sequence[int],
) -> None:
    if not seq or not template:
        raise FrameworkLockError("binder sequence and template are required")
    if len(seq) != len(template):
        raise FrameworkLockError(
            f"length mismatch {len(seq)} vs template {len(template)}"
        )
    allowed = set(AA20)
    bad_chars = sorted({c for c in seq if c not in allowed})
    if bad_chars:
        raise FrameworkLockError(f"non-amino-acid characters in binder: {bad_chars}")
    cdr = set(int(i) for i in cdr_idx0)
    if any(i < 0 or i >= len(seq) for i in cdr):
        raise FrameworkLockError("CDR index outside binder length")
    bad = [
        (i, seq[i], template[i])
        for i in range(len(seq))
        if i not in cdr and seq[i] != template[i]
    ]
    if bad:
        raise FrameworkLockError(
            f"framework mutations at {bad[:10]} (showing up to 10)"
        )


def mutate_cdrs(
    seq: str,
    template: str,
    cdr_idx0: Sequence[int],
    rng,
    n_mut: int,
    omit: str = "C",
) -> str:
    """Propose a CDR-only edit. Raises if the parent sequence already broke the lock."""
    assert_framework_identical(seq, template, cdr_idx0)
    omit_set = set(omit or "")
    alphabet = [aa for aa in AA20 if aa not in omit_set]
    if len(alphabet) < 2:
        raise FrameworkLockError("omit_aas leaves fewer than 2 amino acids")
    positions = [int(i) for i in cdr_idx0]
    if not positions:
        raise FrameworkLockError("no CDR positions to design")
    n = max(1, min(int(n_mut), len(positions)))
    chosen = rng.choice(positions, size=n, replace=False)
    out = list(seq)
    for raw_i in chosen:
        i = int(raw_i)
        choices = [aa for aa in alphabet if aa != out[i]]
        if not choices:
            raise FrameworkLockError(f"no allowed substitution at CDR index {i}")
        out[i] = choices[int(rng.integers(0, len(choices)))]
    proposed = "".join(out)
    assert_framework_identical(proposed, template, cdr_idx0)
    if proposed == seq:
        raise FrameworkLockError("mutation operator produced no CDR change")
    return proposed


def randomize_cdrs(
    template: str,
    cdr_idx0: Sequence[int],
    rng,
    omit: str = "C",
) -> str:
    omit_set = set(omit or "")
    alphabet = [aa for aa in AA20 if aa not in omit_set]
    if not alphabet:
        raise FrameworkLockError("omit_aas removes the entire alphabet")
    out = list(template)
    for i in cdr_idx0:
        out[int(i)] = alphabet[int(rng.integers(0, len(alphabet)))]
    proposed = "".join(out)
    assert_framework_identical(proposed, template, cdr_idx0)
    return proposed
