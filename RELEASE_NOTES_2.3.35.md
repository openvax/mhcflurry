# MHCflurry 2.3.35

Unified PPV@N and ranking-metric calculation (#527):

- Added public `mhcflurry.ranking_metrics` (`ppv_at_n` and `ranking_metrics`) implementing the #405 tie-invariant analytic expected-hits-within-ties rule, and routed all four evaluation call sites through it: `mhcflurry eval compare-models`, `mhcflurry eval paper-figures`, `scripts/training/compare_presentation_percentiles.py`, and `scripts/training/summarize_presentation_affinity_ensemble.py` (along with `processing_affinity_control` and training evaluation scripts that previously imported private `compare_models._metrics`).
- **Metric policy note:** `mhcflurry eval paper-figures` (previously hash-ordered tie breaking) and `scripts/training/summarize_presentation_affinity_ensemble.py` (previously input-row-order tie breaking) now use the same expected-hits-within-ties PPV@N rule as `mhcflurry eval compare-models`, so PPV@N values agree across tables and rendered figures on tied scores.
- Deduplicated `METRIC_DISPLAY_NAMES`, `score_values`, and `path_is_within` into `mhcflurry.cli.model_comparison_constants`.

Model weights, catalogue identities, and predictions are unchanged.
