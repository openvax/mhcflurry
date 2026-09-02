# Experiment snapshots

Generated experiment snapshots live in timestamped directories:

```text
experiments/YYYYMMDDTHHMMSSZ-name-sourcecommit/
```

Create one with:

```bash
mhcflurry train snapshot-experiment \
    --source-dir results/my-run \
    --experiments-dir experiments \
    --name my-run \
    --source-commit "$(git rev-parse HEAD)" \
    --command-file results/my-run/command.sh
```

Each snapshot contains:

- `experiment.json`: schema, UTC capture time, training and collector commits,
  and source archive;
- `environment.json`: Python, platform, GPU, and installed package versions;
- `source_files.csv`: SHA256, size, role, and copy status for every source artifact;
- `external_inputs.csv`: input hashes preserved by runner provenance;
- `data/training_history.csv`: one plot-friendly row per model/fit/epoch;
- `data/models.csv`, `data/fits.csv`, and `data/model_configs.jsonl`;
- `artifacts/`: copied hyperparameters, manifests, metrics, telemetry, and logs;
- held-out `predictions.csv[.bz2]` tables and generated PDF/PNG/SVG figures;
- optional source archive and exact command files.

Held-out prediction tables are copied even when they exceed `--max-copy-mb`, so
performance figures can be regenerated without rerunning inference. On the same
filesystem they are hard-linked into the immutable snapshot to avoid duplicating
multi-gigabyte tables; `source_files.csv` records `storage=hardlink` or `copy`.
Cross-filesystem snapshots fall back to normal copies. Weights and training
tables are hashed but are not duplicated by default. Other files larger than
`--max-copy-mb` remain inventory-only. Do not modify hard-linked source outputs;
preserve the original run until the snapshot and required model archives have
been copied to durable storage.

Generated snapshot directories are ignored by Git; this README is tracked.

## Rebuilding plots

Learning curves can be regenerated from a copied model manifest:

```bash
mhcflurry train plot-loss-curves \
    --selected-dir experiments/<snapshot>/artifacts/<condition>/models.unselected.combined \
    --out experiments/<snapshot>/plots/<condition>-losses
```

Each paired comparison can regenerate its ROC/PR, per-allele-delta,
per-length, and summary-PDF panels without rerunning predictions:

```bash
mhcflurry plot-model-comparison \
    --input experiments/<snapshot>/artifacts/<condition>/comparison-vs-baseline \
    --summary-pdf experiments/<snapshot>/plots/<condition>-comparison.pdf
```

Affinity-factorial runs create one direct, overlap-excluded comparison and one
review PDF per candidate against the pinned public `models.no_additional_ms`
predictor. `benchmark_identity.sha256` in each `affinity/summary.json` proves
that the candidate/public comparisons used identical ordered holdout rows.

For a shortlist, build one shared prediction/score table and a paper-figure
suite containing every finalist plus public 2.2:

```bash
mhcflurry eval affinity-candidate-figures \
    --factorial-dir experiments/<snapshot>/artifacts \
    --condition <candidate-a> \
    --condition <candidate-b> \
    --out experiments/<snapshot>/finalist_figures
```

Canonical external columns already present in the held-out predictions, such as
`netmhcpan4.ba`, `netmhcpan4.el`, and `mixmhcpred`, are retained automatically.
Absent external predictors remain absent rather than being fabricated; they can
first be added with `mhcflurry eval paper-figures external-predictors`.

`artifacts/summary.csv` is the stable optimizer × LSUV × minibatch table for
factorial heatmaps. `data/training_history.csv` is the corresponding tidy
model/fit/epoch table. Publication-style figures and correlation heatmaps can
be regenerated with `mhcflurry eval paper-figures` when a run preserved its
saved score tables.
