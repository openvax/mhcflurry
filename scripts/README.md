# scripts/

Each file here is a long-lived utility. **Anything transient — debugging
runs, one-off parity checks, smoketests, GPU-variant launchers tied to a
single provisioning incident — must not live here.** If it isn't useful
for the next person who clones the repo six months from now, it doesn't
belong.

## Layout

- **`scripts/training/`** — the production training pipeline. Stage scripts
  (`pan_allele_release_*.sh`, `presentation_from_affinity.sh`) and the
  sweep + plot tools that publish artifacts. See its own README.
- **`scripts/dev/`** — developer ergonomic helpers (e.g. relocating
  large run-output dirs outside the rsync source tree). Not invoked by
  CI or release.
- **Top level** — model-validation utilities. Each compares a freshly-
  trained predictor against a baseline (the public release, a fixture,
  or its own pseudosequence loader) and is meant to be re-run by any
  future contributor doing the same diligence.

## Top-level files

- **`validate_against_public.py`** — affinity + presentation rank-correlation
  against the public mhcflurry release on a peptide × allele grid. Quick
  smell test that a new training run hasn't regressed.
- **`validate_allele_sequences.py`** — confirms the
  `allele_sequences.csv → renormalized → pseudosequence` pipeline is
  bit-stable across releases. Catches regressions in the mhcgnomes parse
  layer that would silently misroute predictions.
- **`validate_presentation_with_flanks.py`** — fixed 10-peptide regression
  set including SLLQHLIGL (PRAME) with real flanks; rank correlation vs
  public release. Cheap acceptance test before publishing a new bundle.
- **`modal_train_mhcflurry.py`** — generic Modal launcher; passes any
  supported `mhcflurry-class1-train-*` template through to a Modal
  worker pool. Keeps Modal as a documented training backend, distinct
  from the Brev / local paths the production scripts assume.

## What used to live here (deleted)

- TF→PyTorch parity scripts (`compare_tf_pytorch_random_outputs.py`,
  `cross_allele_parity_analysis.py`, `extract_high_presentation_fixture.py`,
  `plot_fixture_diffs.py`, `generate_fixture_error_report.py`) — finished
  serving the 2.0 → 2.1 migration; not enduring.
- Per-GPU-shape job launchers (`jobs/pan_allele_release_exact_l40s.py`
  etc.) — wrote them during 8×A100 provisioning incidents that have
  since cleared.
- Smoketests of every kind. Tests live in `test/`.
