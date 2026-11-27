# Recorded experiment scripts

Self-contained reproducible scripts for completed validation and ablation
campaigns, separated from the maintained release training stages in
[`scripts/training/`](../training/README.md) and [`scripts/release/`](../release/README.md).

Each subdirectory ties a script set to its published experiment report or
maintainer note so script provenance stays explicit.

## Directory map

### [`release_2_3_0/`](release_2_3_0/)

Reproducible ablation and evaluation campaigns conducted for the `2.3.0` weight
release (see [`docs/release_training_recipe.md`](../../docs/release_training_recipe.md)
and [`notes/README.md`](../../notes/README.md)):

- **Affinity optimizer / LSUV / minibatch factorial** (`test/test_affinity_factorial.py`):
  - `generate_affinity_factorial.py`
  - `run_affinity_factorial.sh`
  - `run_affinity_factorial_architecture_evaluation.sh`
  - `evaluate_affinity_factorial_public.sh`
  - `split_affinity_architectures.py`
  - `summarize_affinity_factorial.py`
  - `summarize_affinity_factorial_architectures.py`
  - `summarize_training_stops.py`
  - `verify_affinity_factorial_models.py`
- **Processing regularization & activation sweep** (`test/test_affinity_factorial.py`):
  - `generate_processing_regularization_activation.py`
  - `run_processing_regularization_activation.sh`
- **Processing flank ablation, fold-ensemble, and subset diagnostics**
  (`test/test_processing_flank_ablation.py`, `test/test_processing_fold_ensembles.py`,
  `test/test_processing_ensemble_subsets.py`):
  - `processing_flank_ablation.py`
  - `evaluate_processing_fold_ensembles.py`
  - `evaluate_processing_subsets.py`

### [`release_2_5_0/`](release_2_5_0/)

- **`benchmark_cross_species_mhcseqs.py`** — End-to-end collection, zero-shot
  evaluation, and figure generation for the `2.5.0` pan-allele `mhcseqs`
  cross-species study published in
  [`docs/cross_species_validation.md`](../../docs/cross_species_validation.md).

## Adding new experiments

- Place reusable operational tools in the `mhcflurry` CLI (`mhcflurry/cli/`) or
  `scripts/training/`.
- Place completed, reproducible study scripts under
  `scripts/experiments/<release_or_study>/` and record their corresponding report
  in this README and [`docs/experiments.md`](../../docs/experiments.md).
- Keep transient local debugging and one-off launchers in the gitignored `jobs/`
  directory rather than committing them.
