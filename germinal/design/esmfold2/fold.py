"""ESMFold2 forward fold. Loaded only when a campaign actually folds.

The installed public API is ESMFold2InputBuilder.fold: diffusion sampling
that returns a structure. This module does not ask that API for sequence
gradients, and it does not pass template, coordinate, pocket, distogram,
or modification inputs. Binder chains are built with msa left unset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from germinal.design.esmfold2.msa import MsaPolicyError, PreparedChain, assert_binder_unpaired


def _numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class EsmFolder:
    def __init__(
        self,
        model,
        builder,
        *,
        num_loops: int,
        num_sampling_steps: int,
        msa_max_depth: int,
    ):
        self.model = model
        self.builder = builder
        self.num_loops = int(num_loops)
        self.num_sampling_steps = int(num_sampling_steps)
        self.msa_max_depth = int(msa_max_depth)
        self._msa_cache = {}

    def protein_inputs(self, chains: list[PreparedChain]):
        from esm.models.esmfold2 import ProteinInput
        from esm.utils.msa.msa import MSA

        assert_binder_unpaired(chains)
        inputs = []
        for chain in chains:
            msa = None
            if chain.role == "target":
                if not chain.msa_path:
                    raise MsaPolicyError(f"target {chain.id} has no MSA at fold time")
                cached = self._msa_cache.get(chain.msa_path)
                if cached is None:
                    cached = MSA.from_a3m(
                        chain.msa_path,
                        remove_insertions=True,
                        max_sequences=4096,
                    )
                    query = "".join(
                        a
                        for a in cached.entries[0].sequence
                        if a.isupper() and a.isalpha()
                    )
                    if query != chain.sequence:
                        raise MsaPolicyError(
                            f"loaded MSA query != target {chain.id} "
                            f"({len(query)} vs {len(chain.sequence)})"
                        )
                    self._msa_cache[chain.msa_path] = cached
                msa = cached
            inputs.append(ProteinInput(id=chain.id, sequence=chain.sequence, msa=msa))
        return inputs

    def __call__(self, chains: list[PreparedChain], seed: int) -> dict:
        from esm.models.esmfold2 import StructurePredictionInput

        spec = StructurePredictionInput(sequences=self.protein_inputs(chains))
        result = self.builder.fold(
            self.model,
            spec,
            num_loops=self.num_loops,
            num_sampling_steps=self.num_sampling_steps,
            num_diffusion_samples=1,
            seed=int(seed),
            msa_max_depth=self.msa_max_depth,
            complex_id=f"esmfold2_s{seed}",
        )
        if isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError(f"expected one diffusion sample, got {len(result)}")
            result = result[0]
        chain_ids = None
        complex_obj = getattr(result, "complex", None)
        if complex_obj is not None and hasattr(complex_obj, "chain_id"):
            raw_ids = complex_obj.chain_id
            if hasattr(raw_ids, "detach"):
                raw_ids = raw_ids.detach().cpu().tolist()
            chain_ids = [str(x) for x in list(raw_ids)]
        iptm = getattr(result, "iptm", None)
        if hasattr(iptm, "detach"):
            iptm = float(iptm.detach().cpu().reshape(-1)[0])
        elif iptm is not None:
            iptm = float(np.asarray(iptm).reshape(-1)[0])
        return {
            "chain_ids": chain_ids,
            "plddt": _numpy(getattr(result, "plddt", None)),
            "pae": _numpy(getattr(result, "pae", None)),
            "iptm_global": iptm,
        }


def load_esmfold2(model_path: str, ccd_cache: str, device: str = "cuda"):
    """Load weights. ccd_cache must be the model directory, not a pickle file."""
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

    ccd = Path(ccd_cache)
    if not ccd.is_dir():
        raise RuntimeError(
            f"BLOCKED_CCD_CACHE_NOT_DIRECTORY: {ccd}. "
            "Pass the model directory. Do not point this at a ccd pickle and "
            "do not let the builder download CCD."
        )
    model = EsmFold2Model.from_pretrained(str(model_path), device=device).eval()
    builder = ESMFold2InputBuilder(ccd_cache=ccd)
    return model, builder
