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
- optional source archive and exact command files.

Weights and training tables are hashed but are not duplicated by default. Files
larger than `--max-copy-mb` also remain inventory-only. Preserve the original
run until the snapshot and any required model archives have been copied to
durable storage.

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

`artifacts/summary.csv` is the stable optimizer × LSUV × minibatch table for
factorial heatmaps. `data/training_history.csv` is the corresponding tidy
model/fit/epoch table. Publication-style figures and correlation heatmaps can
be regenerated with `mhcflurry eval paper-figures` when a run preserved its
saved score tables.
