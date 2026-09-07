# Evaluation-only completion of saved 2.3.0 candidates

## Processing matching policy change: implementation spec

New processing comparisons and new processing training must use the same
sample/length/affinity matching algorithm. Require identical peptide length,
the same sample, and absolute log10 predicted-affinity distance at most 0.25
for every negative; prefer same-protein matches within that bound. Use a
frozen public affinity reference, never the processing candidate's associated
new affinity model. Fail with an actionable error if the candidate pool cannot
supply the requested number of distinct negatives. Do not drop hits, widen the
caliper or fall back to random peptides silently. Evaluation uses ten negatives
per hit; training defaults to one to keep the new data recipe approximately
balanced without multiplying the training budget tenfold. Both ratios are
explicitly recorded and cannot be compared as equal-prevalence AUPRC tasks.

Persist risk-set/source identities, affinities, match ranks and distances,
reference fingerprints, full candidate-pool predictions for joins, and matched
prediction tables used by metrics and plots. Match assignments must not depend
on processing scores or model choice. Training/selection validate the saved
matching contract, including on resume; historical exact-data replay requires
an explicit legacy policy. Inner early-stopping validation groups complete
samples to prevent reused decoys crossing its row split. Public weights and
historical experiment archives remain unchanged and labelled legacy-trained.

Affinity and end-to-end presentation tasks retain their own objectives and
cohorts. Their processing score columns are reusable caches, not unmatched
processing-quality claims. Processing figure paths must clearly carry the
cohort policy and must not put matched processing AUPRC and unmatched
presentation AUPRC on a common comparison axis. No large retraining is
authorized by changing this policy; validate with cached held-out scores and
small regression/smoke tests first.

## Spec and acceptance criteria

Evaluate the completed full train-and-select candidate before spending more
training budget. Do not retrain or modify any source run. Use a separate,
resumable evaluation directory bound to source-model fingerprints, training
provenance, evaluator commit, and frozen holdout. Score presentation first,
then all processing modes, then the train-excluded and descriptive affinity
comparisons. Preserve row-level predictions, per-sample/allele/length metrics,
commands, logs and figure PDFs. No comparison may change the heldout negatives
or silently omit an unavailable component.

The primary with-flank processing model is the actual eight-network hybrid
inside the presentation bundle, not the independently selected legacy model.
Also score the selected legacy ensemble separately. Keep the historical 15-aa
Glorot/Keras and terminal-only affinity provenance exceptions visible.

Optionally finish the already-trained exact-public-data eight-network replay
in a separate output subtree. Verify original data hashes, original fold/row
identity checks, completed training/selection markers and four networks per
family before assembly. Refit identical presentation combiners on the archived
public table. This is a small controlled replay, NOT a full exact-data
train-and-select experiment. Never relabel it as one or use this workflow to
authorize additional-data training automatically.

Commands:

```shell
mhcflurry eval saved-candidate \
  --candidate /persist/runs/COMPLETED_FULL_RUN \
  --exact-processing-run /persist/runs/COMPLETED_EIGHT_NETWORK_RUN \
  --public-root /persist/downloads/2.2.0 \
  --release-holdout-dir /persist/inputs/release_holdout \
  --source-commit EVALUATOR_COMMIT \
  --out /persist/runs/NEW_EVALUATION_RUN
```

The output `candidate/presentation/models` is a portable copy of the saved
candidate bundle. Processing evaluation views use that bundle's actual
components. Nothing is published or automatically accepted for release.
Joint macro AUPRC/PPV@N improvements need micro, sample, length, allele and
locus safeguards; repeated consultation of this holdout makes exploratory
selection intervals optimistic. Independent confirmation and the full
exact-data comparison remain distinct gates.

Run remotely from a clean source archive with the maintained
`launch_saved_candidate_evaluation_modal.py` runplz adapter. It allocates one
A100-40GB, requests 8 CPU cores/64 GiB and has an eight-hour timeout. It has no
training command. Every output stays on the persistent Modal volume even if
the client disconnects. Collect to an existing local parent directory, then
use `mhcflurry train snapshot-experiment` and `mhcflurry eval collate-figures`.

Assembly parent-directory failures are tracked in
[issue #399](https://github.com/openvax/mhcflurry/issues/399).

## Backend and loading failure record

The first evaluation launch (`20260907-saved-candidate-eval-0c638372e`,
Modal app `ap-AFYKcpQzSeHGVwWs0rBUQ9`) received a cancellation signal at
2026-09-07 11:35:56 UTC while loading the presentation benchmark. No completed
comparison or Python error traceback was produced. The cause is not established;
do not call it an OOM or a training failure. Evidence and the missing durable
backend status/collection contract are reported in
[runplz #165](https://github.com/pirl-unc/runplz/issues/165#issuecomment-5570409980).

Independently, [mhcflurry #400](https://github.com/openvax/mhcflurry/issues/400)
tracks the confirmed eager-loading problem: all 76 multiallelic benchmark
files were retained before frozen-holdout filtering. The loader now validates
all inputs in bounded chunks and retains only the frozen samples before
concatenation. Required fields and labels remain checked even on discarded
rows, and file limiting still follows holdout selection. This reduces memory
requirements without changing the evaluation negatives, row order or metrics.

The retry container does not expose cgroup files. Future launches record
per-process RSS as an explicitly labelled fallback, not as a container memory
limit or an OOM counter. The running `b6370ece4` evaluation is not hot-patched;
its memory checks use read-only `ps` queries and its original source identity
is retained.

Uncalibrated exact-data combiners are evaluated explicitly with
`--presentation-score-kinds presentation_score` (issue #401). This preserves
raw scores and does not fabricate percentile ranks or pass percentile release
validation. Default comparisons still require both scores and calibrated
percentiles. Use `--phase exact` (`SAVED_EVAL_PHASE=exact` remotely) to finish
that replay independently, without repeating the full candidate comparisons.

Paired sample intervals can be generated directly from comparison tables:

```shell
mhcflurry eval paired-sample-metrics \
  --metrics COMPARISON/presentation/per_sample_with_flanks_presentation_score.csv \
  --unit-columns sample_id --condition-column model \
  --metric-columns pr_auc ppv_at_n \
  --comparison-labels full-candidate selected-public --baseline selected-public \
  --out EXPERIMENT/paired-presentation-with-flanks
```

## Fixed-weight flank diagnostic: spec

Compare the saved full and public flank/no-flank ensembles on the identical
random-decoy cohort and existing affinity-matched risk sets. Audit flank
completeness and sample/length effects before interpreting the aggregate gap.
These independently selected ensembles are not a controlled flank ablation.

For direct reliance on flanks, use `mhcflurry eval processing-flank-ablation`
on the saved matched table: score each unchanged predictor with real flanks,
both external flanks masked as unknown, and paired N/C flanks shuffled within
sample and peptide length, without conditioning the shuffle on the hit label.
Repeated source rows receive identical perturbations; save the donor row IDs,
seed, model hashes, predictions, per-sample metrics and flank audit. No fitting,
model selection or holdout-label optimization is allowed in this diagnostic.
Masking/shuffling measures sensitivity of already-trained models, not the
performance of an optimally retrained peptide-only model, and can shift the
input distribution. Affinity matching also changes prevalence, so compare
flank/no-flank deltas within each cohort, not absolute AUPRC between cohorts.

Example (local inference; no cloud allocation):

```shell
mhcflurry eval processing-flank-ablation \
  --input CONTROLLED/matched_predictions.csv.bz2 \
  --predictor candidate=CANDIDATE/presentation/models/processing_predictor_with_flanks \
  --predictor public=PUBLIC/models_class1_processing/models.selected.short_flanks \
  --backend cpu --out EXPERIMENT/fixed-weight-flanks
```

Use `--backend mps` where the local Apple GPU is available. For a focused
paired plot from a larger cached screen, `paired-sample-metrics --conditions`
accepts explicit score names and requires the baseline to be included.

The second Modal attempt (`b6370ece4`) completed presentation, all processing
modes and legacy-only processing, then was cancelled during affinity evaluation
at 2026-09-07 14:40:02 UTC. The host traceback shows a DNS/connection failure
while awaiting `runner.remote()`; no agent stop was sent. Durable outputs were
collected. This is stronger evidence for a client-lifetime problem than the
first cancellation; it does not retroactively establish the first cause.
See [runplz #165](https://github.com/pirl-unc/runplz/issues/165#issuecomment-5572663748).

## Flank diagnostic results (2026-09-07)

The first tables below are preserved historical diagnostics. The stricter
re-evaluation at the end of this document supersedes their primary matched
metrics by requiring distinct negative peptide sequences within every risk
set, as well as a hard affinity caliper for all fallback matches.

All values below are sample-macro AUPRC / PPV@N on the same 10 frozen samples.
These are actual selected ensemble weights, not architecture-only proxies.

| Cohort | New hybrid 5-aa | New no-flank | Public 5-aa | Public no-flank |
| --- | --- | --- | --- | --- |
| Original random decoys | 0.2375 / 0.3183 | 0.2142 / 0.3085 | 0.2202 / 0.2995 | 0.2279 / 0.3173 |
| Affinity/length-matched decoys | 0.3945 / 0.4284 | 0.3470 / 0.3949 | 0.3971 / 0.4284 | 0.3506 / 0.3981 |

The original cohort has 2,054,263 rows and 18,507 hits. The matched cohort has
203,577 rows (one hit and ten decoys per risk set), with reused decoys and
143,879 unique source rows. Same sample/length is required; same-protein decoys
are preferred within a log10-affinity caliper of 0.25, but only 28.1% of decoys
are same-protein matches. Median affinity distance is 0.000606 log10 units;
the 95th percentile is 0.1896. This controls predicted affinity, not measured
binding, and does not fully control protein origin.

The new flank-vs-no-flank AUPRC gain is +0.04751 (paired sample bootstrap
95% interval +0.03785 to +0.05637; 10/10 samples improve). PPV@N gain is
+0.03348 (+0.02226 to +0.04530; 9/10 improve). These exploratory intervals
condition on the selected weights and do not correct for prior model search.
Different decoy prevalence prevents interpreting the higher matched AUPRC
itself as a performance gain.

Fixed-weight counterfactual inference (local MPS, seed 42):

| External context | New hybrid AUPRC / PPV@N | Public 5-aa AUPRC / PPV@N |
| --- | --- | --- |
| Real | 0.3945 / 0.4285 | 0.3971 / 0.4284 |
| Masked to unknown | 0.3233 / 0.3771 | 0.2224 / 0.2715 |
| Shuffled within sample/length | 0.2878 / 0.3307 | 0.2639 / 0.3149 |

All six inference conditions use unchanged weights. Real-context MPS scores
differ from archived CUDA scores by at most 0.000214 (new) / 0.000314 (public);
the slight rounding/tie differences above are not new fits. Masking and
shuffling both shift the input distribution. Shuffling preserves paired N/C
flanks and is independent of hit labels; repeated decoys use the same donor.

The data and code support three conclusions:

1. Flanks contain useful information and these ensembles actively rely on it.
2. No-flank models still see the entire peptide, including residues on the
   peptide side of both cleavage sites and the length-dependent pooling path.
   The target is ligand-vs-decoy discrimination, not direct cleavage labels.
   Historical training selected strong predicted binders, as in the original
   [MHCflurry 2.0 study](https://pubmed.ncbi.nlm.nih.gov/32711842/); this reduces
   binding confounding but does not make processing a pure protease assay.
3. The original benchmark mixes affinity, length and processing discrimination.
   Nine-mers are 68.9% of hits but about 25% of decoys. Even before affinity
   matching, within 9-mers new 5-aa versus new no-flank gives macro AUPRC
   0.3959 versus 0.3453. Matching changes both length composition and affinity;
   do not attribute the whole change exclusively to affinity.

Missing context is not the main explanation: among unique matched source rows,
97.58% / 97.03% of hits and 99.10% / 99.26% of decoys have complete standard-AA
local N/C contexts. This is a completeness audit, not proof of correct protein
or isoform assignment. The new hybrid's boundary members use context dropout
0.25, but no dropout-only causal comparison was performed here.

Do not accept the changed-data candidate yet: it ties public processing under
affinity control, the standalone new no-flank and 15-aa ensembles regress,
and presentation-percentile AUPRC regresses despite better raw presentation
scores ([issue #402](https://github.com/openvax/mhcflurry/issues/402)). Full
affinity evaluation and exact-data replay evaluation remain unfinished.

## Strict matching-policy validation (2026-09-07)

The reusable matcher now caps **every** hit/decoy affinity difference at 0.25
log10 units and permits each negative peptide sequence only once per risk set.
It is shared by new training, evaluation and cached-data validation; see
[issue #403](https://github.com/openvax/mhcflurry/issues/403).
Rebuilt risk sets retain all 18,507 hits across the same ten frozen samples
(203,577 rows). No new weights were trained for this re-evaluation.

| Saved ensemble | Macro AUPRC | Macro PPV@N |
| --- | ---: | ---: |
| New hybrid, 4 new legacy-CNN + 4 new boundary-CNN networks | 0.395247 | 0.428876 |
| Public flank | 0.397445 | 0.428589 |
| New no-flank | 0.347097 | 0.394391 |
| Public no-flank | 0.351409 | 0.398763 |

The older matched table contained 1,064 repeated negative peptide entries
within risk sets (a sequence could appear through different protein mappings).
The revised matcher removes those duplicates without dropping hits; 7,630
assignment-row positions change, including rank shifts. The maximum affinity
distance is 0.249999. These slightly revised metrics supersede the original
matched ensemble table above, not the original weight/provenance records.

Against public flank weights, the new hybrid's AUPRC difference is -0.002199
(95% paired sample interval -0.010675 to +0.006749); PPV@N is +0.000287
(-0.006837 to +0.007734). It does not pass the joint improvement gate.
The hybrid contains **no reused public processing weights**: "legacy" names
the architecture of four newly trained members. Its four boundary members
also retain a whole-peptide CNN trunk. The fixed family balance was a
diversity hedge, not evidence that four plus four is optimal.

The next controlled comparison should train CNN-only and boundary candidates
on the same newly matched data/folds/seeds, then select and compare CNN-only,
boundary-only and freely mixed ensembles. Keep public weights as the fixed
reference and retain per-sample/micro safeguards. Do not launch another full
grid or claim matched-training gains from this cached evaluation.

Reproduce with `output/processing-matching-policy-20260907/reproduce.sh`.
The output directory retains the canonical full-cohort scores, matched
predictions, per-sample and per-length metrics, paired bootstrap draws,
input hashes and figure inputs. A timestamped `experiments/` snapshot preserves
this revision independently of the earlier diagnostic archive.
