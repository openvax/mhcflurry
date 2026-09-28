# Processing preparation acceleration

## Workflow

Optimize the maintained `mhcflurry train processing-data` path without changing
the scientific matching policy or interrupting the existing frozen Modal run.

1. Factor a bounded sample pipeline with one scoring owner. CPU stages prepare
   the next sample and commit/match the previous sample while the caller scores
   the current sample. Each sample is a resumable coroutine yielding score
   requests; CPU continuations save immutable rounds before matching or requesting
   expansion. Bound active samples and CPU workers, propagate errors, drain started
   writes, and publish completion only after durable artifacts and their hashes.
   Independent per-sample RNGs make scheduling irrelevant to candidate selection.
2. Replace per-hit matching loops with grouped, chunked batched nearest-neighbor
   lookup. Preserve the existing sorted local search window, stable tie order,
   same-protein priority, global fallback exclusions, caliper inclusivity, failure
   identities and output row order. Never allocate a hits-by-entire-pool matrix.
   Keep a scalar oracle in tests and compare randomized/adversarial cases and a
   saved real scored pool before measuring speed.
3. Introduce a numeric protein/window representation with an explicit device
   gather API. Encode each protein once; keep position, length and peptide identity
   numeric during sampling/deduplication/exclusion and affinity input preparation.
   Materialize peptide/flank strings for the persisted scored artifacts. Integrate
   the numeric route with the actual affinity scoring API, not a disconnected GPU
   demonstration. Preserve historical window bounds and model input encodings;
   test CPU and available accelerator equivalence, padding, exclusions and seeds.

Verification includes pipeline overlap/backpressure/error/interrupt tests,
checkpoint resume and corruption regressions, scalar/vector matching parity,
numeric/string encoding and prediction parity, deterministic end-to-end sample
preparation, lint, the full test suite, and reproducible benchmark JSON. Saved
real-data checks distinguish exact matching parity from floating-point prediction
tolerances. No unmeasured end-to-end speedup or new model-quality claim.

## APIs and execution

- `scoring_pipeline(workflows, score, max_in_flight=3, cpu_workers=2)` runs CPU
  coroutine continuations around a single caller-owned scorer. It bounds submitted
  work as well as worker count. A scored round commits before any matching,
  expansion or completion; failures cancel queued work and drain running writes.
- `sample_preparation_steps(...)` is the resumable sample coroutine;
  `prepare_sample(...)` remains its synchronous adapter. `ScoredValues` carries
  actual inference duration separately from queue waiting. Round metadata also
  records string materialization and artifact-writing times.
- `SortedAffinityPools` packs sample/length/protein groups into stable numeric
  search indices. The normal path has no per-hit or per-protein Python lookup.
  Exceptional missing-identity and exhausted-window behavior retains the scalar
  semantics, including pandas' distinctions among missing-value representations.
- `ProcessingProteome` encodes proteins once per worker. `ProteinWindows` retains
  sampled positions and collision-free numeric identities. `NumericCandidatePool`
  removes duplicate scoring inputs and delays string construction until export.
- `NumericSequences.aligned_tensor(...)` and
  `Class1AffinityPredictor.predict_numeric(..., allele=...)` gather/align numeric
  inputs on CPU, MPS or CUDA and reuse the existing neural networks and log-affinity
  ensemble aggregation. The numeric API is deliberately single-allele and strict
  about unsupported lengths/residues, matching preparation's monoallelic contract.

The maintained `mhcflurry train processing-data` command defaults to pipeline
depth 3 (two CPU continuations and one scoring owner per process). Set
`--preparation-pipeline-depth 1` for a serial-stage control. Processes receive
small sample chunks so preparation can overlap within a worker; each sample's
seed still derives only from the master seed and sample ID. Chunking/scheduling
does not change candidate draws or output order. The explicit historical
`legacy-top-binders` recipe remains serial and uses its old sampler.

Numeric sampling still chooses/filter positions on CPU; device-side gathering,
alignment and prediction are implemented. This does not claim the entire pipeline
is CUDA-only. CPU preparation does not own a GPU context. The encoded proteome's
device copy is created lazily by the sole scoring owner.

## Reproducible measurements

```bash
mhcflurry train benchmark-processing-preparation \
  --out benchmark.json --repeats 5 \
  --scored-pool original.candidate_pool.csv.bz2 \
  --scored-pool expansion.round-01.csv.bz2 \
  --baseline-matching-source frozen-source/mhcflurry/processing_matching.py
```

Add `--affinity-predictor /path/to/models.combined` to check actual ensemble
prediction equivalence and timings. This benchmark fingerprints inputs, reference
source/weights and implementation files, verifies exact sampling/export and
matching/diagnostic parity, and reports all timing repetitions. Sampler data are
synthetic; supplied scored matching pools can be real. Inference timing excludes
initial loading and device staging after warming both paths. CPU/MPS tests cover
alignment and ordinary/merged neural ensembles; CUDA tests run when available.

The initial grouped prototype did not improve whole-matcher runtime. Profiling
identified redundant peptide-string sorting in overlap checks, in addition to
per-group work. Both were replaced with packed numeric indices. The saved real
389,641-row A3303 pool then matched in median 0.509 seconds versus 1.906 seconds
for the frozen reference (five repetitions; 3.74x), with identical 5,790 output
rows and diagnostics. These are component timings, not a measured full-pipeline
or A100 throughput gain. Keep the unsuccessful initial benchmark as well.

A later shared-machine rerun reversed the timing result: 22.35 s vectorized
versus 14.34 s baseline (0.64x), still with exact parity. Both implementations
were much slower than before, and separate timing blocks could be biased by
changing load. Thus 3.74x is an observed initial result, not an established
speedup for the final implementation. The benchmark now alternates comparison
order and records CPU time as well as wall time; retain both favorable and
unfavorable runs and measure throughput on the target machine before deployment.

The alternating-order final-source check (four repetitions) measured median
0.744 s vectorized versus 2.239 s scalar (3.01x); median process CPU time was
0.518 s versus 1.657 s. Every paired repetition favored the vectorized matcher,
and assignments/diagnostics were exact. This supports a component gain despite
the shared-host variability, not a full-pipeline or A100 throughput claim.

Actual public 2.2 pan-allele weights (ten networks, fingerprint
`82e5950e9570e88f76ea9f42893bbc1f4eee0895ef952db70dedf1fdb0ae7898`)
produced exactly equal numeric/string predictions for 1,000 synthetic peptides
at HLA-A*02:01 on CPU. Median warmed inference was 0.680 s numeric and 0.673 s
string across three repetitions: no CPU inference speedup. Numeric representation
defers export cost and enables device gathering; it does not eliminate the cost
of saving sequence strings. These timings were taken on a shared development
machine while regression checks were running.

Tracking: [#403](https://github.com/openvax/mhcflurry/issues/403), draft PR
[#362](https://github.com/openvax/mhcflurry/pull/362). The already-running Modal
kernel sweep retains its frozen source and is not restarted for these changes.
