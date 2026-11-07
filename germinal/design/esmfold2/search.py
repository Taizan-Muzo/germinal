"""Discrete CDR search scored by an injected fold function.

Each proposal is a framework-locked CDR edit. The fold function is the
structure model. A candidate whose binder pLDDT or binder–target ipSAE is
missing or non-finite is rejected and cannot become the incumbent. Global
iPTM is ignored by the accept/reject rule.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from germinal.design.esmfold2.framework import (
    assert_framework_identical,
    mutate_cdrs,
)
from germinal.design.esmfold2.msa import PreparedChain, assert_binder_unpaired
from germinal.design.esmfold2.scoring import evaluate_complex


FoldFn = Callable[[list[PreparedChain], int], dict]


def _with_binder(chains: list[PreparedChain], binder_sequence: str) -> list[PreparedChain]:
    out: list[PreparedChain] = []
    replaced = False
    for chain in chains:
        if chain.role == "binder":
            if chain.msa_path is not None:
                raise RuntimeError("binder MSA leaked into a search step")
            out.append(
                PreparedChain(
                    id=chain.id,
                    sequence=binder_sequence,
                    role="binder",
                    msa_path=None,
                    msa_query_sha256=None,
                )
            )
            replaced = True
        else:
            out.append(chain)
    if not replaced:
        raise RuntimeError("search chains have no binder")
    assert_binder_unpaired(out)
    return out


def _score_step(
    chains: list[PreparedChain],
    *,
    expected: list[tuple[str, str, int]],
    fold_fn: FoldFn,
    seed: int,
    pae_cutoff: float,
    min_binder_plddt: float | None,
    min_binder_ipsae: float | None,
) -> dict:
    raw = fold_fn(chains, seed)
    scored = evaluate_complex(
        chain_ids=raw.get("chain_ids"),
        plddt=raw.get("plddt"),
        pae=raw.get("pae"),
        expected=expected,
        iptm_global=raw.get("iptm_global"),
        pae_cutoff=pae_cutoff,
        min_binder_plddt=min_binder_plddt,
        min_binder_ipsae=min_binder_ipsae,
    )
    scored["fold_seed"] = int(seed)
    return scored


def run_search(
    *,
    parent_chains: list[PreparedChain],
    template: str,
    cdr_idx0: list[int],
    expected: list[tuple[str, str, int]],
    fold_fn: FoldFn,
    rng: np.random.Generator,
    steps: int,
    mutations_per_step: int,
    omit_aas: str,
    temperature_start: float,
    temperature_end: float,
    base_seed: int,
    pae_cutoff: float = 10.0,
    min_binder_plddt: float | None = None,
    min_binder_ipsae: float | None = None,
    propose: Callable[[str, np.random.Generator], str] | None = None,
) -> dict:
    """Hill-climb / anneal CDR sequences. Returns the campaign record."""
    binder = next(c.sequence for c in parent_chains if c.role == "binder")
    assert_framework_identical(binder, template, cdr_idx0)
    assert_binder_unpaired(parent_chains)

    def propose_default(seq: str, local_rng: np.random.Generator) -> str:
        return mutate_cdrs(
            seq, template, cdr_idx0, local_rng, mutations_per_step, omit_aas
        )

    proposer = propose or propose_default
    trajectory = []

    def consider(seq: str, step: int, kind: str) -> dict:
        assert_framework_identical(seq, template, cdr_idx0)
        chains = _with_binder(parent_chains, seq)
        scored = _score_step(
            chains,
            expected=expected,
            fold_fn=fold_fn,
            seed=base_seed + step + 1,
            pae_cutoff=pae_cutoff,
            min_binder_plddt=min_binder_plddt,
            min_binder_ipsae=min_binder_ipsae,
        )
        row = {
            "step": step,
            "kind": kind,
            "binder_sequence": seq,
            "status": scored["status"],
            "reason": scored["reason"],
            "objective": scored["objective"],
            "binder_plddt": scored["binder_plddt"],
            "binder_target_ipsae": scored["binder_target_ipsae"],
            "global_iptm": scored["global_iptm"],
            "global_iptm_used_for_acceptance": False,
            "passes_thresholds": scored["passes_thresholds"],
        }
        trajectory.append(row)
        return scored

    current_seq = binder
    current = consider(current_seq, 0, "initial")
    best = current if current["status"] == "SCORED" else None
    best_seq = current_seq if best else None

    n_steps = int(steps)
    for step in range(1, n_steps + 1):
        proposed = proposer(current_seq, rng)
        assert_framework_identical(proposed, template, cdr_idx0)
        scored = consider(proposed, step, "proposal")
        if scored["status"] != "SCORED":
            trajectory[-1]["decision"] = "reject_fail_closed"
            continue
        accept = False
        if current["status"] != "SCORED":
            accept = True
        else:
            delta = float(scored["objective"]) - float(current["objective"])
            if delta >= 0:
                accept = True
            else:
                span = max(n_steps - 1, 1)
                temp = temperature_start + (temperature_end - temperature_start) * (
                    (step - 1) / span
                )
                temp = max(float(temp), 1e-8)
                accept = float(rng.random()) < math.exp(delta / temp)
        trajectory[-1]["decision"] = "accept" if accept else "reject"
        if accept:
            current_seq = proposed
            current = scored
            if best is None or float(scored["objective"]) > float(best["objective"]):
                best = scored
                best_seq = proposed

    return {
        "generator": "esmfold2_discrete_cdr_search",
        "structure_model": "esmfold2",
        "differentiable_logits": False,
        "af2_called": False,
        "trajectory": trajectory,
        "best_sequence": best_seq,
        "best": best,
        "final_sequence": current_seq if current["status"] == "SCORED" else None,
    }
