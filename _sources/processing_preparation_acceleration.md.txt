# Processing data preparation

`mhcflurry train processing-data` prepares affinity-matched negative peptides
with bounded CPU/scoring overlap and resumable, checksummed artifacts. The
matching contract is described in {doc}`probabilistic_processing_matching`.

## Execution and saved artifacts

Each sample is a resumable workflow. CPU stages sample candidate windows and
save scored rounds while a single scoring owner per process runs the affinity
predictor. The default pipeline depth is three; use
`--preparation-pipeline-depth 1` for serial stages. Sample seeds derive from the
master seed and sample identity, so scheduling does not change candidate draws.

`ProcessingProteome` encodes proteins once per worker. `ProteinWindows` keeps
sampled positions and peptide identities numeric; `NumericCandidatePool`
deduplicates scoring inputs before peptide/flank strings are materialized for
saved tables. `NumericSequences.aligned_tensor()` and
`Class1AffinityPredictor.predict_numeric(..., allele=...)` gather and align
inputs on CPU, MPS or CUDA. This strict, single-allele API uses the existing
networks and log-affinity ensemble aggregation. Sampling remains on the CPU.

Each scored round is saved before matching or expansion. An insufficient pool
triggers additional draws for unresolved peptide lengths, within the configured
round limit. Completion is recorded only after matched outputs and their hashes
are saved. Errors retain committed rounds; they do not publish a completed sample.

Use `--resume` for the same output or `--resume-matching-dir PRIOR.matching`
when creating a new output. Resume checks input/reference hashes, matching policy,
seed, observed rows and saved scores. It never silently reuses old assignments
from a different matching policy. The explicit `legacy-top-binders` recipe uses
the historical serial preparation path.

## Benchmarking

```bash
mhcflurry train benchmark-processing-preparation \
  --out benchmark.json --repeats 5 \
  --scored-pool sample.round-00.csv.bz2 \
  --scored-pool sample.round-01.csv.bz2
```

Supply all rounds needed for a feasible matching pool. Add
`--affinity-predictor /path/to/models.combined` to compare actual numeric/string
predictions and timings. Outputs record input and source hashes, runtime versions,
all timing repetitions and parity checks. Sampling inputs are synthetic; supplied
scored pools can be real. Inference timings exclude loading and initial staging.

`--baseline-matching-source /path/to/processing_matching.py` compares two
implementations of the **same matching policy** and requires exact assignment and
diagnostic equality. A nearest-neighbor matcher that reuses negatives is not an
appropriate parity baseline for the current without-replacement policy.

Component timings do not establish an end-to-end preparation speedup or a gain
in trained-model accuracy. The tests separately cover pipeline backpressure and
errors, resume integrity, matching invariants, and numeric/string prediction parity.
