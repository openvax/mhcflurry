# Release training recipe

The 2.3.0 weights use the `final-2.3.0-candidate-v2` recipe identifier and the
`current` data vintage: the curated 2023 affinity snapshot plus the configured
mass-spectrometry sources. The identifier is fixed for reproducibility; its
name does not describe the software's release status. The machine-readable
settings are in `scripts/training/final_230_candidate_v2_recipe.json`.

## Components

| Component | Release configuration |
|---|---|
| Affinity | 35 architectures × 4 folds; minibatch 1024; native PyTorch RMSprop; Glorot initialization; pre-activation LSUV; dropout 0.5; patience 20; terminal weights for selection. |
| Processing without flanks | 128 architectures × 4 folds; minibatch 512; Glorot initialization and Keras-equation Adam; selected ensemble. |
| Short-flank processing | One kernel-13 architecture × 4 folds; five residues on each side; native RMSprop; Glorot initialization; best inner-validation AP checkpoints. |
| Boundary processing | One large-ReLU architecture × 4 folds; five flank residues and five peptide residues at each cleavage boundary; Glorot initialization, Keras-equation Adam and terminal checkpoints. |
| Presentation with flanks | Equal mixture of the four short-flank and four boundary networks, combined with affinity by the fitted logistic model. |
| Presentation without flanks | Selected no-flank processing ensemble, combined with affinity by its separately fitted logistic model. |
| Long-flank processing | Separate 15-residue-flank diagnostic grid; it is not the processing ensemble embedded in the presentation predictor. |

Training checkpoints and model-selection records remain in the training run.
An inference download needs the selected weights, model manifests, allele
sequences, fitted presentation coefficients and calibration. Training data and
provenance are retained for overlap audits; unused alternative checkpoints are
not part of the inference model.

## Data and evaluation

Affinity uses measured affinities and the configured mass-spec reassignment.
Its synthetic negatives are random amino-acid peptides. Processing training
uses protein-derived negatives matched within sample, length and affinity
constraints. Presentation training uses protein-derived negatives in its own
training table. These training samplers do not define a comparison's test set.

A model comparison must freeze one positive/negative row set, score every
comparator on it, and exclude training overlap identically for every model.
Sample/source disjointness and peptide disjointness are separate checks;
missing lineage cannot be treated as proof of separation. The maintained
release holdout manifests are an input exclusion mechanism, not a guarantee
that an arbitrary public comparator has never seen the same data.

Report end-to-end presentation, affinity-only ranking and processing-only
ranking separately. Never average their AP values. For multi-patient results,
report per-patient metrics and paired patient-bootstrap intervals. See
{doc}`evaluation` for score orientation, common-row joins and output formats.

## Calibration and compatibility

Raw affinity is in nM, lower is stronger. Processing and presentation scores
range from zero to one, higher is stronger. Percentile ranks range from zero
to 100, lower is stronger; they are background ranks, not calibrated biological
probabilities.

New calibrations use the shared compact method: 64 knots, increasing to 128
only when independent, label-free background validation justifies it.
Affinity uses log(IC50); processing and presentation use logit(score).
Existing public histogram calibrations load without refitting or conversion.
See {doc}`shared_percent_rank_transforms` for the exact rules and compatibility.

The release workflow calibrates affinity and presentation. Standalone
processing calibration is a separate command with an explicit background and
flank policy. Evaluating raw scores does not validate a percentile API: check
both saved score representations before publishing weights.

## Reproduce a run

Use the maintained release entry point and explicit recipe/data settings;
see the [release workflow](https://github.com/openvax/mhcflurry/tree/master/scripts/release)
and {doc}`final_230_candidate_experiment` for launch and collection commands.
Freeze the source commit, seed, data hashes and complete settings before
training. Preserve the resulting `release_provenance.json`, holdout manifests,
training tables, selection records and comparison outputs.

Worker counts and prediction chunks may be autosized. Training minibatches
are scientific settings and must not be silently reduced to fit a device.
The release path uses eager execution and highest float32 matmul precision.
