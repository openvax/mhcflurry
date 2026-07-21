# Configuration and performance

MHCflurry chooses safe defaults for ordinary prediction and training commands.
Most users should start without environment-variable overrides or fixed worker
counts, then pin a value only when reproducing a benchmark or diagnosing a
specific machine.

## Runtime defaults

`MHCFLURRY_DEFAULT_CLASS1_MODELS`
: Path to the default model directory. If unset, MHCflurry uses the models from
  the installed download bundle.

`MHCFLURRY_OPTIMIZATION_LEVEL`
: Controls pan-allele ensemble merging. The default, `1`, enables the faster
  merged-network path. Set it to `0` when debugging or comparing the unmerged
  implementation.

`MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE`
: Overrides the default prediction batch size. The normal value is `auto`,
  which sizes batches from available device memory and can retry with a smaller
  batch after an allocator out-of-memory error. A fixed integer disables that
  resizing behavior and should be used only when you know the workload fits.

## Automatic hardware planning

Prediction, training, calibration, selection, and evaluation commands share a
workload-aware planner. Depending on the command, it considers available GPU
memory, model and data size, the configured training batch, host RAM, CPU count,
and remaining work items. Container memory limits are detected through cgroups,
so a job does not size itself from RAM that belongs to the physical host but is
unavailable inside the container. On spawn-based platforms, the planner also
measures the loaded worker context before starting the pool; this prevents a
compressed input file from hiding the much larger in-memory copy created in
each worker. If the remaining safe RAM cannot hold even one additional spawn
copy, an automatic plan runs serially in the already-loaded parent process.

The common automatic options are:

| Option | Automatic behavior |
|---|---|
| `--gpus` | Uses every visible CUDA device unless a count is supplied. |
| `--num-jobs auto` | Chooses the total worker count from device and host capacity. |
| `--max-workers-per-gpu auto` | Packs complete estimated worker working sets into each GPU. |
| `--dataloader-num-workers auto` | Sizes pretraining DataLoader children from CPU and host memory. |
| `--torch-compile auto` | Enables compilation only on devices where it is expected to help. |

Automatic GPU packing has no default fixed worker cap. Explicit CLI values and
explicit per-workload memory estimates remain authoritative; the planner logs
the facts and limits used for each decision.

If a workload still runs out of memory, first keep the batch and worker settings
on `auto`. If you need to intervene, reduce `--max-workers-per-gpu` or use a
smaller explicit batch. Pinning values can improve repeatability on a known
machine, but it also bypasses some automatic safeguards.

See {doc}`orchestrator` for implementation details and expert environment
overrides.

## Reproducible training

Training and calibration commands accept `--random-seed`. The command-line
default is `42`, covering data splits and shuffles, initial weights, random
negatives, allele sampling, and calibration peptides. Ensemble members derive
distinct sub-seeds from that master seed.

The direct Python training APIs use `seed=None` by default, preserving their
historical stochastic behavior unless the caller opts in.

## Unified and historical command names

MHCflurry 2.3.0 groups commands under one `mhcflurry` entry point. For example:

```text
mhcflurry-predict                       = mhcflurry predict
mhcflurry-downloads fetch              = mhcflurry downloads fetch
mhcflurry-class1-train-pan-allele-models = mhcflurry class1-train-pan-allele-models
```

Both forms invoke the same implementation. New automation can use the unified
form; existing scripts do not need to change.
