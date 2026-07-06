# scripts/release/

Maintainer helpers for retraining, evaluating, packaging, and publishing model
artifacts.

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
    --affinity-max-workers-per-gpu auto \
    --deploy-mode dry-run
```

Supported backends are:

- `local`: train in the current checkout on the current machine.
- `brev-existing`: train on a named existing Brev instance. Pass
  `--brev-instance NAME`; a missing instance is an error.
- `brev-provision`: train on a named Brev instance, provisioning it if it does
  not already exist. Pass `--brev-instance NAME` to choose the name, or omit it
  to let the script generate one from the release and timestamp. The default
  cleanup policy is `--brev-on-finish stop`; use `leave` for interactive
  debugging or `delete` for disposable runs. The wrapper, not runplz, owns the
  final lifecycle: it verifies the remote exit status, syncs
  `/root/runplz-latest/out` back into `--run-dir`, and only then applies the
  cleanup policy. If `stop` reports success but Brev still shows the provisioned
  instance as `RUNNING`, the default `--brev-stop-failure-action delete` removes
  it after artifacts have synced; use `warn` to keep the instance instead.
  The default `--brev-sync-mode release` copies only release/evaluation inputs
  and telemetry: final selected model directories, runplz events, training logs,
  GPU occupancy, model-comparison outputs, release plots, affinity eval/loss
  plots, and generated configs. Use
  `--brev-sync-mode full` only when you deliberately need every unselected
  candidate model and intermediate CSV for a deep post-mortem.
  Provisioned full-training runs default to the known 4xA100 type
  `a2-highgpu-4g:nvidia-tesla-a100:4`; `--brev-instance-type TYPE` overrides it.
- `ssh`: train on a specific remote host, with `--remote`, `--remote-repo`, and
  `--remote-run-dir`. Authentication comes from local `ssh` / `rsync`
  configuration, typically SSH keys or an SSH config `Host`.

The script runs training, `mhcflurry compare-models`,
`mhcflurry plot-model-comparison`, and deployment validation in order; each
stage has a `--skip-*` flag for resuming. For Brev backends, the expensive
comparison and plot steps run on the remote GPU machine before artifact sync and
cleanup, then the local wrapper uses the synced `eval_comparison/` outputs
instead of repeating release-scale inference on the laptop. Per-step
stdout/stderr logs and a `status.tsv` file are written under
`<run-dir>/workflow_logs/`, alongside the training logs copied from the remote
run (`.runplz/`, `gpu_occupancy.csv`, release driver logs, and
model-selection/evaluation artifacts).

Training batch-size knobs are first-class release options. `--minibatch-size`
sets the shared default (currently 1024); `--affinity-minibatch-size` and
`--processing-minibatch-size` override individual model families. Affinity
training defaults to `--affinity-max-workers-per-gpu auto`, so the training
command estimates per-worker VRAM from the hyperparameter grid, row count, and
minibatch before choosing GPU worker packing; pass an integer only to pin a
known-good worker count for a specific machine. Processing variants default to
`with_flanks no_flank short_flanks`, and the presentation stage uses
`with_flanks` as its with-flank processing predictor unless
`--presentation-processing-with-flanks-kind` says otherwise.

Evaluation now includes affinity, processing, and presentation by default. In
addition to the detailed component CSV/JSON files and plots, `compare-models`
writes `release_summary.csv` and `release_summary.md` with the release-gate
tables used to compare newly trained weights against public weights.
