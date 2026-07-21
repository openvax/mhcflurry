# Maintained scripts

This directory contains reusable training, release, validation, and developer
utilities. Transient experiments, incident-specific launchers, and smoke tests
belong in the ignored `jobs/` directory or in the test suite.

## Directory map

- **`training/`** — production training stages, remote transport, sweeps, and
  profiling. See [training/README.md](training/README.md).
- **`release/`** — end-to-end release orchestration, provenance validation,
  packaging, and deployment. See [release/README.md](release/README.md).
- **`dev/`** — developer ergonomics that are not invoked by CI or releases.
- **Top-level Python files** — focused validation tools for comparing a newly
  trained predictor with a release or fixture.

The maintained public workflow for training, evaluation, plots, remote cleanup,
and optional deployment is:

```shell
mhcflurry train pan-allele-release --help
```

## Validation tools

- **`validate_against_public.py`** checks affinity and presentation rank
  correlation against a public release on a peptide-by-allele grid.
- **`validate_allele_sequences.py`** verifies that shipped pseudosequences are
  stable across model bundles.
- **`validate_presentation_with_flanks.py`** runs a small fixed presentation
  regression set with real flanks.

These are acceptance checks for model work. General behavior tests belong in
`test/`.

## Where new work belongs

- Use ignored `jobs/` for local launchers, interrupted experiments, and
  machine-specific debugging.
- Promote reusable operational tools from `jobs/` into `scripts/`.
- Put reproducible user/release artifacts under
  `downloads-generation/<download_name>/` with a `GENERATE.sh` entry point.
