# ESMFold2 sequence design on Tencent `/run/ti`

This is the production generator. It folds with ESMFold2 and edits CDR
sequences in discrete steps. It does not call the ColabDesign / AF2-Multimer
generator, and a missing GPU or weight directory stops the run.

The public ESMFold2 `fold` API is a diffusion sampler. It returns a
structure, not sequence-logit gradients, so the design loop is not a
ColabDesign logits optimization with the model name swapped.

## Do not reset the dirty tree

`/run/ti/src/germinal` on A800, nginx-router, and JF is a dirty checkout of
`1e1c1a5` plus deployed scientific fixes (framework hard-lock, chain
mapping, fail-closed filters). Do not `git reset`, `git checkout`, or
replace that directory. Do not start the legacy GPU worker or the
supervisors that launch it. Do not delete `SCIENCE_FIX_FREEZE` or
`PRODUCTION_DISABLED`.

Install this branch beside that tree:

```bash
git clone --branch cursor/esmfold2-design-backend-ca28 \
  https://github.com/Taizan-Muzo/germinal.git \
  /run/ti/src/germinal-esmfold2
```

Use the ESMFold2 environment and weights that were already verified:

- Python: `/run/ti/envs/esmfold2/bin/python`
- Weights: `/run/ti/models/esmfold2`
- `ccd_cache` is that same directory. It is not a `ccd.pkl` file, and the
  builder must not download CCD.

## ESMFold2 chain ids

ESM 3.4.2 writes `MolecularComplex.chain_id` as int64 `asym_id`
values, `0, 1, 2, ...` in `ProteinInput` order. The letters passed as
`ProteinInput(id=...)` (`A`, `B`, `C`, `D`) are stored in
`metadata.chain_lookup[asym_id]`, not in `chain_id`. Comparing the raw
integers to the campaign ids produces `chain_id_set_mismatch` and a
fail-closed score.

The scorer applies that lookup and then computes binder pLDDT and
binder–target ipSAE on the campaign ids. A missing lookup, a blank
name, two asym ids sharing one name, or an asym id absent from the
lookup stays `FAIL_CLOSED`. The code does not rename `0` to `A` by
position.

## Pull this checkout and restart only these workers

The running workers loaded the previous code. Pull the esmfold2
checkout and restart those processes. Do not reset or checkout
`/run/ti/src/germinal`.

```bash
cd /run/ti/src/germinal-esmfold2
git fetch origin cursor/esmfold2-design-backend-ca28
git pull --ff-only origin cursor/esmfold2-design-backend-ca28

# Processes of this generator only. Leave every other PID alone.
pgrep -af run_germinal_esmfold2.py
```

Stop those PIDs, then launch the same one-GPU worker command again.
The flock file is `esmfold2_<host>_gpu<index>.lock`. Do not clear
freeze locks and do not start the legacy supervisor. A worker that is
still inside a fold keeps the old scorer until that process exits.

## One worker per GPU

Each process owns one card. Do not pass a comma-separated device list, and
do not put two workers on one card (the older TNF slot split does not apply
here). The worker takes a flock on `esmfold2_<host>_gpu<index>.lock` and
refuses the card if `nvidia-smi` already shows a compute process. It does
not kill that process.

```bash
# GPU 0 only. A second card is a second process, not a second slot.
bash /run/ti/src/germinal-esmfold2/deploy/tencent/gpu_worker_esmfold2.sh \
  0 configs/esmfold2/tnfa_official_monomer_design.yaml

bash /run/ti/src/germinal-esmfold2/deploy/tencent/gpu_worker_esmfold2.sh \
  1 configs/esmfold2/tfr1_official_monomer_design.yaml
```

`CUDA_VISIBLE_DEVICES` is set to that single index. JF0 is not assumed free
because its VRAM is empty; if a compute process or an existing reservation
is on the card, the worker exits and leaves it alone.

## What the campaign does

- Design runs on a monomer: `tnfa_official_monomer` (157 aa, UniProt P01375
  Val77–Leu233) or `tfr1_official_monomer` (672 aa, P02786 Cys89–Phe760).
- Validation is a separate fold: `tnfa_official_trimer` (three copies of the
  same 157 aa sequence, one shared MSA) or `tfr1_official_dimer`. Design
  mode refuses those oligomers.
- The 152 aa TNF PDB construct (`tnfa_design152_monomer`, official sequence
  without N-terminal VRSSS, Asp kept at the UniProt 219 site) needs its own
  MSA. The 157 aa MSA is not sliced. The old 142 aa diagnostic truncation
  is not a construct.
- Every target chain gets a real a3m whose query equals that chain. The
  binder is passed with no MSA. A configured binder MSA is an error.
- Framework positions outside the CDRs (`fw` 25/17/38/14, `cdr` 11/8/18)
  must match `binder_template`. Put a 131-residue parent in the yaml before
  launching. This repo does not invent that parent.
- Hotspot strings are written into `campaign.json` and are not passed as
  coordinates, templates, pockets, or restraints.
- Acceptance uses binder pLDDT and binder–target ipSAE
  (`mean_targets_of_mean_directions_v1`: mean of both PAE directions on each
  binder–target chain, then the mean across target chains). Global iPTM is
  stored and is not an acceptance score.
- A missing chain map, missing PAE, or non-finite score is `FAIL_CLOSED`.
  It cannot be written to `accepted.json`. Set `min_binder_plddt` and
  `min_binder_ipsae` before treating a run as gated; unset thresholds write
  `best_finite.json` only.
- `num_loops: 20` and `num_sampling_steps: 200` match the installed builder
  defaults used in the diagnostic audit. They are not a published contest
  protocol.

`accepted.json` is a finite score that cleared the thresholds you set. It
is not a binding measurement.

## Smoke before any campaign

```bash
cd /run/ti/src/germinal-esmfold2
/run/ti/envs/esmfold2/bin/python scripts/smoke_esmfold2_backend.py \
  /run/ti/mmdesign_like_vhh/runs/esmfold2_smoke
```

Exit 2 writes `SMOKE_STATUS.json`. On a machine with no GPU the status is
`BLOCKED_NO_GPU`. Missing weights are `BLOCKED_NO_MODEL`. A CCD path that is
not a directory is `BLOCKED_CCD_CACHE_NOT_DIRECTORY`. Do not start the
worker when smoke is blocked, and do not point the config at another model
to get past the block.

While freeze locks exist, the worker exits without folding. To run this
worker only:

```bash
ESMFOLD2_CAMPAIGN_UNLOCK=1 bash deploy/tencent/gpu_worker_esmfold2.sh \
  0 configs/esmfold2/tnfa_official_monomer_design.yaml
```

That does not remove the lock and does not start the legacy supervisor.

Check a config without loading weights:

```bash
/run/ti/envs/esmfold2/bin/python run_germinal_esmfold2.py \
  --config configs/esmfold2/tnfa_official_monomer_design.yaml \
  --check-config
```

Fill `binder_template` first. `--check-config` checks the framework lock and
that each target MSA query matches the chain. It does not fold.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | A finite best exists. `accepted.json` is written only when thresholds are set and passed. |
| 2 | Blocked (no GPU, weights, CCD directory, or ESMFold2 package). No design started. |
| 3 | Every fold failed closed, or the best finite score missed a configured threshold. |
| 4 | Config or GPU-selection error (oligomer design, MSA mismatch, missing template, more than one GPU). |
