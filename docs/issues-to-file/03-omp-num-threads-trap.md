# Training recipe: GENERATE.sh relies on OMP_NUM_THREADS=1 but it's not obvious from the code

## Observation

The public release recipe `models_class1_pan/GENERATE.sh` sets at line 60:

```bash
export OMP_NUM_THREADS=1
```

Without this, numpy/MKL inside each mhcflurry worker multi-threads across
all available CPU cores. With N parallel workers (one per GPU on a
multi-GPU box), you get N × cores threads fighting for cores = extreme
oversubscription.

## Impact (reproduced)

On Brev `denvr_A100_sxm4x8` (120 vCPUs, 8 A100s), running the release-exact
GENERATE.sh port **without** `OMP_NUM_THREADS=1`:

- Load average: **713** (6× oversubscription)
- Per-epoch wall time: **200–450 s** for 1024×512 dense (should be <10 s on A100)
- GPU memory utilization: **1–5%** across all 8 GPUs
- ~9 networks trained in 11 hours vs expected ~20–40 networks

After adding `OMP_NUM_THREADS=1` to the recipe, the bottleneck disappears.

## Suggestion

This should either:

1. **Be set inside mhcflurry's training code** when it spawns workers
   (`cluster_parallelism.py` or `class1_train_pan_allele_models.py`), so
   users running a raw `mhcflurry-class1-train-pan-allele-models` without
   the recipe wrapper still benefit.

2. **Be prominently documented** in the training docs / README. It's
   currently just a line in the release GENERATE.sh that you'd miss if
   porting the recipe or writing a new harness.

Option (1) is probably safer — any sharp edge that the public recipe's
author learned the hard way should be encoded in the library, not left to
rediscovery.
