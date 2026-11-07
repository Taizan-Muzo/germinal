"""ESMFold2 sequence-design backend.

The public ESMFold2 builder folds a complex by diffusion sampling. It does
not expose ColabDesign-style sequence-logit gradients, so this package
designs CDR sequences by discrete search and scores every candidate with
an ESMFold2 forward fold.

Nothing in this package imports ColabDesign, JAX, or AF2-Multimer.
"""

__all__ = ["framework", "msa", "scoring", "constructs", "search", "campaign"]
