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

Direct deployment defaults to `--processing-variants "no_flank with_flanks"`.
Pass `--processing-variants "no_flank with_flanks short_flanks"` only when the
short-flank model belongs to the release being packaged. The end-to-end release
wrapper forwards its current variant selection automatically, so leftover model
directories from an earlier run are never included merely because they exist.

The script writes the tarballs, `SHA256SUMS`, and a `downloads.yml` snippet under
`<run-dir>/release-assets/` by default. After upload, commit the corresponding
`mhcflurry/downloads.yml` update in the package release PR.

## End-to-end release workflow

```bash
mhcflurry train pan-allele-release \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --backend local \
    --release-profile full \
    --minibatch-size 1024 \
    --affinity-max-workers-per-gpu auto
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
  Provisioned full-training runs default to automatic selection from current
  Brev inventory using the launcher's built-in resource requirements (4x A100,
  at least 35 GB VRAM per GPU, 32 CPUs, 300 GB RAM, and 1 TB disk). No resource
  environment overrides are required. Use `--brev-provider gcp`, `denvr`, or
  `denvr-80gb` for the common pinned release shapes, or
  `--brev-instance-type TYPE` for any exact Brev type. An exact type takes
  precedence over automatic selection. For auto-selection,
  `--brev-exclude-providers PREFIXES` can remove provider/type prefixes such as
  `oci` from the candidate list.
- `ssh`: train on a specific remote host, with `--remote`, `--remote-repo`, and
  `--remote-run-dir`. Authentication comes from local `ssh` / `rsync`
  configuration, typically SSH keys or an SSH config `Host`.

The command runs training, `mhcflurry eval compare-models`, and
`mhcflurry eval plot-comparison` in order; each training/evaluation/plot stage
has a `--skip-*` flag for resuming. Deployment is opt-in: pass
`--deploy-mode dry-run`, `draft`, or `publish` only when you want the
model-artifact release step to run. For Brev backends, the expensive
comparison and plot steps run on the remote GPU machine before artifact sync and
cleanup, then the local wrapper uses the synced `eval_comparison/` outputs
instead of repeating release-scale inference on the laptop. Per-step
stdout/stderr logs and a `status.tsv` file are written under
`<run-dir>/workflow_logs/`, alongside the training logs copied from the remote
run (`.runplz/`, `gpu_occupancy.csv`, release driver logs, and
model-selection/evaluation artifacts).

The public interface is `mhcflurry train pan-allele-release`. Its current
implementation delegates to `retrain_evaluate_deploy.sh`, an internal process
orchestration engine for shell training stages, traps, remote lifecycle,
telemetry, and artifact synchronization.

`--release-profile full` is the default and trains the complete processing
artifact set (`with_flanks no_flank short_flanks`). Use
`--release-profile fast-8xa100` for throughput runs on 8xA100 / 80 GB machines:
with `--backend brev-provision`, it requests the Denvr 8xA100 80 GB shape when
no provider/type was explicitly supplied. Worker packing still uses
`--affinity-max-workers-per-gpu auto`; pass an explicit value such as
`--affinity-max-workers-per-gpu 2` only after validating that it fits and
improves throughput for that run. Use `--release-profile minimal-processing`
when short-flanks
processing artifacts are intentionally out of scope; it trains and evaluates
only `with_flanks` and `no_flank`. `--release-profile fast-minimal` combines
both. These profiles are opt-in so the default release path keeps the full
artifact contract on the configured provider.

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
tables used to compare newly trained weights against a configurable baseline.
The release wrapper defaults to `--compare-baseline public:2.0.0`, the closest
older public model bundle available in `downloads.yml`, and labels the new side
as `MHCflurry <release>` with any `rcN` suffix stripped. Use
`--compare-baseline public` for the currently configured public release, or pass
another training-run directory / `public:<release_name>` plus
`--compare-baseline-label` to tune the figure labels.
`eval plot-comparison` now writes both release-diagnostic plots and a
paper-style `plots/paper/` suite: per-allele affinity scatter panels,
per-sample processing/presentation scatter panels, per-length bars, delta
boxplots, and release-summary overview panels. The release wrapper also asks it
to write vector-first `plots/model_comparison_figures.pdf` so remote runs sync a
portable publication-review packet without a separate local plotting step.

For the broader paper-style figure suite, pass
`--paper-figures-scores-dir /path/to/saved/evaluation/outputs` (or set
`PAPER_FIGURES_SCORES_DIR`). That directory may contain derived score tables
such as `accuracy_scores.multiallelic.csv`, saved test-set prediction tables
such as `benchmark.multiallelic.csv.bz2`, or optional metadata/artwork tables.
You can also pass explicit saved prediction tables with
`--paper-figures-multiallelic-predictions` and
`--paper-figures-monoallelic-predictions`. The legacy
`--paper-figures-artifacts-dir` / `PAPER_FIGURES_ARTIFACTS_DIR` spelling is
still accepted as an alias for older 2023 bundle paths. The wrapper passes
those inputs through to `mhcflurry eval plot-comparison`, which invokes
`mhcflurry eval paper-figures render` and writes SVG/PDF/PNG panels plus
`paper_figures.pdf`, `manifest.csv`, and `missing_inputs.md` under
`eval_comparison/plots/paper_figures/`. Figure families whose source tables are
absent are reported there instead of being faked. The comparator suite is
configurable through `mhcflurry eval paper-figures render` flags such as
`--candidate-predictor`, `--external-baselines`, and `--preferred-predictors`;
the defaults use NetMHCpan 4.0 BA/EL and MixMHCpred when those saved
prediction columns are available.

For a local model-to-figures pass outside the release wrapper, use
`mhcflurry eval paper-figures run --a RUN_DIR --b public --out RUN_DIR/eval`.
That command composes compare-models, paper-figures, and the diagnostic plot
PDF. Use `mhcflurry eval paper-figures score-predictions` to precompute
`accuracy_scores.*.csv` caches from canonical saved prediction tables. The
workflow does not run external predictors itself; external NetMHCpan /
MixMHCpred outputs should be generated separately and passed as numeric columns
in those canonical prediction tables. Score direction is explicit: built-in
predictor names have defaults, and custom score columns should be described in
`predictor_info.csv` with `predictor` and `higher_is_better` columns (or passed
to `score-predictions --predictor-info`). See the
[evaluation artifact map](../../docs/commandline_tools.md#evaluation-and-plotting-artifacts)
for the full contract.

For Brev-backed training, local paper inputs stay on the control machine by
default. Remote MHCflurry evaluation runs on the GPU instance, syncs back, and
then the wrapper renders paper figures locally if `--paper-figures-scores-dir`
or explicit saved prediction tables were supplied. This is the intended path
when NetMHCpan/MixMHCpred are installed locally but not on the Brev image. To
prepare those local inputs during the remote training window, pass
`--paper-figures-prepare-command "..."`; the command runs in the background on
the control machine and must write to the directory/file paths also supplied via
`--paper-figures-scores-dir`, `--paper-figures-multiallelic-predictions`, or
`--paper-figures-monoallelic-predictions`. If you use `mhctools` for external
predictors, call it through
`mhcflurry eval paper-figures external-predictors`: `mhctools` depends on
MHCflurry, so MHCflurry does not import it or list it as a dependency.

Example:

```bash
PAPER_BENCHMARK="$PWD/notebooks/2023-retraining/artifacts/benchmark.multiallelic.csv.bz2"

mhcflurry train pan-allele-release \
    --run-dir runs/2.3.0 \
    --release 2.3.0 \
    --backend brev-provision \
    --paper-figures-scores-dir "$PWD/runs/2.3.0/external_predictions" \
    --paper-figures-external-baselines "netmhcpan4.2.ba,netmhcpan4.2.el,mixmhcpred" \
    --paper-figures-prepare-command \
        "mhcflurry eval paper-figures external-predictors \
            --input '$PAPER_BENCHMARK' \
            --out '$PWD/runs/2.3.0/external_predictions/benchmark.multiallelic.csv.bz2' \
            --predictor netmhcpan42-ba:netmhcpan4.2.ba:affinity \
            --predictor netmhcpan42-el:netmhcpan4.2.el:score \
            --predictor mixmhcpred:mixmhcpred:score"
```

The same release workflow is also available from the unified CLI:

```bash
mhcflurry train pan-allele-release \
    --run-dir /path/to/release-run \
    --release 2.3.0 \
    --backend brev-provision
```
