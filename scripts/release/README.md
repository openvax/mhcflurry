# scripts/release/

Maintainer helpers for packaging and publishing trained model artifacts. These
scripts operate on completed training runs; they do not train or evaluate models.

## Model deployment

```bash
scripts/release/deploy_trained_models.sh \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --github-release 2.3.0 \
    --mode dry-run
```

Use `--mode draft` to build archives and upload them to a draft GitHub release.
Use `--mode publish` only after the GitHub release already exists; this script
does not publish the release itself because publishing also triggers package
release workflows.

The script writes the tarballs, `SHA256SUMS`, and a `downloads.yml` snippet under
`<run-dir>/release-assets/` by default. After upload, commit the corresponding
`mhcflurry/downloads.yml` update in the package release PR.

## End-to-end release workflow

```bash
scripts/release/retrain_evaluate_deploy.sh \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --backend local \
    --minibatch-size 1024 \
    --deploy-mode dry-run
```

Supported backends are:

- `local`: train in the current checkout on the current machine.
- `brev-existing`: train on existing Brev/runplz capacity. This does not
  provision a new machine; runplz/Brev handles remote execution, package sync,
  and credentials.
- `ssh`: train on a specific remote host, with `--remote`, `--remote-repo`, and
  `--remote-run-dir`. Authentication comes from local `ssh` / `rsync`
  configuration, typically SSH keys or an SSH config `Host`.

The script runs training, `mhcflurry compare-models`,
`mhcflurry plot-model-comparison`, and deployment validation in order; each
stage has a `--skip-*` flag for resuming.

Training batch-size knobs are first-class release options. `--minibatch-size`
sets the shared default (currently 1024); `--affinity-minibatch-size` and
`--processing-minibatch-size` override individual model families. Processing
variants default to `with_flanks no_flank short_flanks`, and the presentation
stage uses `with_flanks` as its with-flank processing predictor unless
`--presentation-processing-with-flanks-kind` says otherwise.

Evaluation now includes affinity, processing, and presentation by default. In
addition to the detailed component CSV/JSON files and plots, `compare-models`
writes `release_summary.csv` and `release_summary.md` with the release-gate
tables used to compare newly trained weights against public weights.
