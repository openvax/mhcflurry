# Maintained scripts

This directory separates reusable operational scripts from recorded experiment
campaigns. Transient debugging and machine-specific launchers belong in the
gitignored `jobs/` directory.

## Directory map

- **`training/`** — active release-training stages, remote transport, and
  shared helpers (`gpu_telemetry.sh`, `set_cpu_threads.sh`). See
  [training/README.md](training/README.md).
- **`release/`** — end-to-end release orchestration
  (`retrain_evaluate_deploy.sh`), provenance validation, packaging, and
  deployment. See [release/README.md](release/README.md).
- **`experiments/`** — self-contained reproducible scripts for completed
  validation and ablation studies (`release_2_3_0/`, `release_2_5_0/`), each
  tied to its published doc or note. See
  [experiments/README.md](experiments/README.md).
- **Top-level Python files** — focused CI and acceptance utilities:
  - `classify_ci_changes.py` — `#471` fail-safe CI change classifier.
  - `validate_against_public.py` — affinity and presentation rank correlation
    against a public release on a peptide-by-allele grid.
  - `validate_allele_sequences.py` — verifies shipped pseudosequences are
    stable across model bundles.
  - `validate_presentation_with_flanks.py` — fixed presentation regression set
    with real flanks.

Packaged CLI workflows live under `mhcflurry train --help` and
`mhcflurry eval --help`; full end-to-end release orchestration is driven by
`scripts/release/retrain_evaluate_deploy.sh --help`.

## Where new work belongs

- Promote reusable user/operator workflows into `mhcflurry/cli/` (`mhcflurry
  train` / `mhcflurry eval`) rather than adding scattered top-level scripts.
- Record completed reproducible studies under `scripts/experiments/<study>/` with
  a clear link in `scripts/experiments/README.md` and `docs/experiments.md`.
- Keep transient local debugging and interrupted runs in gitignored `jobs/`.
- `downloads-generation/` retains historical dataset/bundle recipes only.
