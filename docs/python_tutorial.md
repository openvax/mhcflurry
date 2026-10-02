# Python library tutorial

For most applications, use
{class}`~mhcflurry.Class1PresentationPredictor`: it returns binding, processing,
and combined presentation predictions from one interface. Use the lower-level
affinity or processing predictors only when you need those components by
themselves.

## Loading a predictor

`Class1PresentationPredictor.load()` without a path uses the downloaded
release model (see {ref}`downloading`); pass a model directory to load a custom
predictor.

```{doctest}
>>> from mhcflurry import Class1PresentationPredictor
>>> predictor = Class1PresentationPredictor.load()
>>> "HLA-A*02:01" in predictor.supported_alleles
True
```

## Predicting for individual peptides

{meth}`~mhcflurry.Class1PresentationPredictor.predict` returns a
{class}`pandas.DataFrame` with binding affinity, processing, and presentation
predictions, whose values depend on the loaded model release:

```{doctest}
>>> predictions = predictor.predict(
...     peptides=["TPVCPNGPG", "RLLEGMEMI"],
...     n_flanks=["MSSSS", "MVENK"],
...     c_flanks=["NCQV", "FGQVI"],
...     alleles=["HLA-A0201", "HLA-A0301"],
...     verbose=0)
>>> predictions.peptide.tolist()
['TPVCPNGPG', 'RLLEGMEMI']
>>> bool(predictions.presentation_score.between(0, 1).all())
True
```

The peptides and flanks above come from `example.fasta`. Omit both
`n_flanks` and `c_flanks` to compare the same peptides without source context.

Here, the allele list is one MHC class I allele set, and the
strongest binder across that MHC allele set is reported for each peptide.

| Python input | Meaning |
|---|---|
| `alleles=["A0201", "A0301"]` | One MHC allele set; one result per peptide. |
| `alleles={"sample1": [...], "sample2": [...]}` | Multiple named MHC allele sets; one result per sample and peptide. |

```{note}
MHCflurry normalizes allele names using the [mhcgnomes](https://github.com/pirl-unc/mhcgnomes)
package. Names like `HLA-A0201` or `A*02:01` will be
normalized to `HLA-A*02:01`, so most naming conventions can be used
with methods such as {meth}`~mhcflurry.Class1PresentationPredictor.predict`.
```

Invalid, ambiguous, class-II, pseudogene, null, and unsupported allele names
raise a descriptive `ValueError` by default. For streaming or mixed-quality
data, pass `throw=False`; affected prediction rows are retained with `NaN`
scores (or ignored when another valid allele in the MHC allele set supplies the
sample's best affinity) while valid inputs are still evaluated. The
command-line equivalent is `mhcflurry predict --no-throw`.

If you have multiple sample MHC allele sets, you can pass a dict, where the
keys are arbitrary sample names:

```{doctest}
>>> predictions = predictor.predict(
...     peptides=["KSEYMTSWFY", "NLVPMVATV"],
...     alleles={
...        "sample1": ["A0201", "A0301", "B0702", "B4402", "C0201", "C0702"],
...        "sample2": ["A0101", "A0206", "B5701", "C0202"],
...     },
...     verbose=0)
>>> list(zip(predictions.sample_name, predictions.peptide))
[('sample1', 'KSEYMTSWFY'), ('sample1', 'NLVPMVATV'), ('sample2', 'KSEYMTSWFY'), ('sample2', 'NLVPMVATV')]
```

Here the strongest binder for each sample / peptide pair is returned.

## Scanning protein sequences

{meth}`~mhcflurry.Class1PresentationPredictor.predict_sequences` scans protein
sequences for MHC ligands. This example keeps 8–11mers with a predicted binding
affinity of at most 500 nM to any allele in either of two samples:

```{doctest}
>>> scan = predictor.predict_sequences(
...    sequences={
...        'protein1': "MDSKGSSQKGSRLLLLLVVSNLL",
...        'protein2': "SSLPTPEDKEQAQQTHH",
...    },
...    alleles={
...        "sample1": ["A0201", "A0301", "B0702"],
...        "sample2": ["A0101", "C0202"],
...    },
...    result="filtered",
...    comparison_quantity="affinity",
...    filter_value=500,
...    verbose=0)
>>> bool(len(scan) > 0 and scan.affinity.le(500).all())
True
>>> {"sequence_name", "peptide", "best_allele"}.issubset(scan.columns)
True
```

When using `predict_sequences`, the flanking sequences for each peptide are
automatically included in the processing and presentation predictions.

## Lower level interfaces

The {class}`~mhcflurry.Class1PresentationPredictor` delegates to a
{class}`~mhcflurry.Class1AffinityPredictor` instance for binding affinity predictions.
If you only need binding affinities, use this instance directly:

```{doctest}
>>> affinity_predictor = predictor.affinity_predictor
>>> affinities = affinity_predictor.predict_to_dataframe(
...     allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])
>>> affinities[["peptide", "allele"]].to_dict("records")
[{'peptide': 'SIINFEKL', 'allele': 'HLA-A*02:01'}, {'peptide': 'SIINFEQL', 'allele': 'HLA-A*02:01'}]
>>> bool(((affinities.prediction_low < affinities.prediction) &
...       (affinities.prediction < affinities.prediction_high)).all())
True
```

Alternatively, `Class1AffinityPredictor.load()` selects the active release's
standalone affinity bundle when installed, falling back to the affinity
component of its presentation bundle. Accessing `predictor.affinity_predictor`
as above guarantees that both calls use the same loaded affinity ensemble.

The affinity predictor treats alleles per peptide rather than as an MHC allele
set:

| Python input | Meaning |
|---|---|
| `predict_to_dataframe(allele="A0201", ...)` | Score every peptide against one allele. |
| `predict_to_dataframe(alleles=[...], ...)` | Pair each peptide with the allele at the same position. |

The `prediction_low` and `prediction_high` fields give the 5-95 percentile
predictions across the models in the ensemble. This detailed information is not
available through the higher-level {class}`~mhcflurry.Class1PresentationPredictor`
interface.

Under the hood, `Class1AffinityPredictor` itself delegates to an ensemble of
{class}`~mhcflurry.Class1NeuralNetwork` instances, which implement the neural network
models used for prediction. To fit your own models, start with the
{doc}`training` guide; {meth}`~mhcflurry.Class1NeuralNetwork.fit` is the
underlying Python method.

You can similarly use {class}`~mhcflurry.Class1ProcessingPredictor` directly for
antigen processing prediction, and there is a low-level
{class}`~mhcflurry.Class1ProcessingNeuralNetwork` with a {meth}`~mhcflurry.Class1ProcessingNeuralNetwork.fit` method.

When interpreting percentile outputs, lower means stronger and loading a
model preserves its saved calibration. For custom calibration or standalone
processing percentiles, see {doc}`shared_percent_rank_transforms`.
