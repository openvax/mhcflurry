# Release workflows

Maintainer tools for retraining, evaluating, synchronizing, packaging, and
publishing model artifacts. Prediction users do not need these scripts.

## End-to-end workflow

Use the public command rather than invoking the orchestration shell script
directly:

```shell
mhcflurry train pan-allele-release \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --backend local
```

The workflow runs these stages in order:

1. Train affinity, processing, and presentation models.
2. Compare the new models with a configured public release or run directory.
3. Render diagnostic plots and a combined PDF.
4. Copy remote artifacts back when using a remote backend.
5. Optionally package or deploy model archives.

Each stage has a `--skip-*` option for controlled resumption. Logs and
`status.tsv` are written under `<run-dir>/workflow_logs/`. Deployment is off by
default.

The command delegates process orchestration to
`retrain_evaluate_deploy.sh`. That script remains internal because it owns shell
training stages, traps, remote lifecycle, telemetry, and artifact
synchronization.

## Choose a backend

| Backend | Use it when | Required options |
|---|---|---|
| `local` | The current machine has the required compute and storage. | None beyond the common arguments. |
| `brev-existing` | A named Brev instance already exists. | `--brev-instance NAME` |
| `brev-provision` | The workflow should select or create a Brev instance. | None; the name and shape can be automatic. |
| `ssh` | Training should run in a specific remote checkout. | `--remote`, `--remote-repo`, `--remote-run-dir` |

SSH authentication comes from the local `ssh` and `rsync` configuration. Before
training, the workflow verifies the remote commit and tracked worktree so model
provenance cannot be stamped from a different checkout.

## Brev selection, synchronization, and cleanup

`brev-provision` defaults to automatic inventory selection using the release
workload requirements: 4 A100-class GPUs with at least 35 GB VRAM each, 32 CPUs,
300 GB RAM, and 1 TB disk. No resource environment variables are required.

Use one of these options only when the default selection is unsuitable:

- `--brev-provider gcp`, `denvr`, or `denvr-80gb` selects a common release
  shape.
- `--brev-instance-type TYPE` selects an exact Brev type and takes precedence
  over automatic provider selection.
- `--brev-exclude-providers PREFIXES` removes provider/type prefixes from
  automatic candidates.

The wrapper verifies remote completion and copies artifacts before changing the
instance state. The default `--brev-on-finish stop` stops a successful or failed
run after synchronization. Use `delete` for disposable instances or `leave`
only for active debugging. If a provisioned instance remains running after a
successful stop request, the default stop-failure policy deletes it after
artifacts are safe locally.

`--brev-sync-mode release` copies the artifacts needed for evaluation and
publication: selected models, logs, events, telemetry, comparison prediction
tables and summaries, plots, and generated configuration. This keeps a synced
run sufficient for a later plot-only resume after the remote instance is
cleaned up. Use `--brev-sync-mode full` only for a deliberate post-mortem that
needs every candidate model and training intermediate table.

## Release profiles and performance

`--release-profile full` is the default. It trains the complete processing set:
`with_flanks`, `no_flank`, and `short_flanks`. Presentation uses the true
`with_flanks` predictor by default.

Optional profiles are:

- `fast-8xa100`: selects the common 8×A100 80 GB Brev shape when no machine
  override was supplied.
- `minimal-processing`: omits `short_flanks` from both training and evaluation.
- `fast-minimal`: combines those two choices.

The shared training minibatch defaults to 1024. Use
`--affinity-minibatch-size` or `--processing-minibatch-size` only when a model
family needs a different value.

Affinity worker packing defaults to `--affinity-max-workers-per-gpu auto`. The
training command estimates the complete per-worker working set from the model,
data, configured batch, and detected memory. Pin an integer only after measuring
a known machine; explicit values bypass automatic packing decisions.

## Evaluation and figures

Evaluation covers affinity, processing, and presentation. The workflow writes
component metrics, `release_summary.csv`, `release_summary.md`, individual
plots, and `plots/model_comparison_figures.pdf`. Its default baseline is
`public:2.0.0`; use `--compare-baseline public` for the currently configured
public release or provide another run directory/version.

On Brev, comparison and diagnostic plotting run on the GPU instance before
synchronization. This avoids repeating release-scale inference on a laptop.

Broader paper figures can use saved prediction tables or score caches through
`--paper-figures-scores-dir` and the explicit
`--paper-figures-*-predictions` options. Local licensed predictors may be
prepared while remote training runs with `--paper-figures-prepare-command`;
their inputs remain on the control machine and render after remote artifacts
arrive.

See the [evaluation guide](../../docs/evaluation.md) for the canonical output
map, saved-prediction schema, score orientation, and external-predictor adapter.
Run `mhcflurry train pan-allele-release --help` for the complete release option
list.

## Package and deploy models

Package a completed run without uploading it:

```shell
scripts/release/deploy_trained_models.sh \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --github-release 2.3.0 \
    --mode dry-run
```

The script writes archives, `SHA256SUMS`, and a `downloads.yml` snippet under
`<run-dir>/release-assets/` by default.

- `--mode draft` uploads to a draft GitHub release.
- `--mode publish` uploads only after the GitHub release exists; it does not
  publish that release because publication also triggers package workflows.

Direct deployment packages `no_flank` and `with_flanks` by default. Include
`short_flanks` only when it belongs to the current release. The end-to-end
workflow forwards its exact trained variant set, so stale directories from an
older run are never packaged merely because they exist.
