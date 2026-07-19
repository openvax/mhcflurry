# Python library tutorial

The MHCflurry Python API exposes additional options and features beyond those
supported by the commandline tools and can be more convenient for interactive
analyses and bioinformatic pipelines. This tutorial gives a basic overview
of the most important functionality. See the {ref}`API-documentation` for further
details.

## Loading a predictor

Most prediction tasks can be performed using the
{class}`~mhcflurry.Class1PresentationPredictor` class, which provides a programmatic API
to the functionality in the {ref}`mhcflurry-predict <ref-mhcflurry-predict>` and
{ref}`mhcflurry-predict-scan <ref-mhcflurry-predict-scan>` commands.

Instances of {class}`~mhcflurry.Class1PresentationPredictor` wrap a
{class}`~mhcflurry.Class1AffinityPredictor` to generate binding affinity predictions
and a {class}`~mhcflurry.Class1ProcessingPredictor` to generate antigen processing
predictions. The presentation score is computed using a logistic regression
model over binding affinity and processing predictions.

Use the {meth}`~mhcflurry.Class1PresentationPredictor.load` static method to load a
trained predictor from disk. With no arguments this method will load the predictor
released with MHCflurry (see {ref}`downloading`). If you pass a path to a
models directory, then it will load that predictor instead.

```{doctest}
>>> from mhcflurry import Class1PresentationPredictor
>>> predictor = Class1PresentationPredictor.load()
>>> predictor.supported_alleles[:5]
['Atbe-B*01:01', 'Atbe-E*03:01', 'Atbe-G*03:01', 'Atbe-G*03:02', 'Atbe-G*06:01']
```

## Predicting for individual peptides

To generate predictions for individual peptides, we can use the
{meth}`~mhcflurry.Class1PresentationPredictor.predict` method of the {class}`~mhcflurry.Class1PresentationPredictor`,
loaded above. This method returns a {class}`pandas.DataFrame` with binding affinity, processing, and presentation
predictions:

```{doctest}
>>> predictions = predictor.predict(
...     peptides=["SIINFEKL", "NLVPMVATV"],
...     alleles=["HLA-A0201", "HLA-A0301"],
...     verbose=0)
>>> [(row.peptide, str(row.best_allele)) for row in predictions.itertuples()]
[('SIINFEKL', 'HLA-A0201'), ('NLVPMVATV', 'HLA-A0201')]
>>> predictions.presentation_score.round(2).tolist()
[0.02, 0.97]
```

Here, the list of alleles is taken to be an individual's MHC I genotype (i.e. up
to 6 alleles), and the strongest binder across alleles for each peptide is
reported.

```{note}
MHCflurry normalizes allele names using the [mhcgnomes](https://github.com/til-unc/mhcgnomes)
package. Names like `HLA-A0201` or `A*02:01` will be
normalized to `HLA-A*02:01`, so most naming conventions can be used
with methods such as {meth}`~mhcflurry.Class1PresentationPredictor.predict`.
```

If you have multiple sample genotypes, you can pass a dict, where the
keys are arbitrary sample names:

```{doctest}
>>> predictions = predictor.predict(
...     peptides=["KSEYMTSWFY", "NLVPMVATV"],
...     alleles={
...        "sample1": ["A0201", "A0301", "B0702", "B4402", "C0201", "C0702"],
...        "sample2": ["A0101", "A0206", "B5701", "C0202"],
...     },
...     verbose=0)
>>> [(row.sample_name, row.peptide, str(row.best_allele))
...  for row in predictions.itertuples()]
[('sample1', 'KSEYMTSWFY', 'C0201'), ('sample1', 'NLVPMVATV', 'A0201'), ('sample2', 'KSEYMTSWFY', 'A0101'), ('sample2', 'NLVPMVATV', 'A0206')]
```

Here the strongest binder for each sample / peptide pair is returned.

Processing and presentation predictions can use the upstream (N-flank) and
downstream (C-flank) sequence from the source protein for better cleavage
context. Pass those sequences with `n_flanks` and `c_flanks`:

```{doctest}
>>> predictions = predictor.predict(
...     peptides=["KSEYMTSWFY", "NLVPMVATV"],
...     n_flanks=["NNNNNNN", "SSSSSSSS"],
...     c_flanks=["CCCCCCCC", "YYYAAAA"],
...     alleles={
...        "sample1": ["A0201", "A0301", "B0702", "B4402", "C0201", "C0702"],
...        "sample2": ["A0101", "A0206", "B5701", "C0202"],
...     },
...     verbose=0)
>>> predictions[["n_flank", "c_flank"]].drop_duplicates().to_dict("records")
[{'n_flank': 'NNNNNNN', 'c_flank': 'CCCCCCCC'}, {'n_flank': 'SSSSSSSS', 'c_flank': 'YYYAAAA'}]
>>> predictions.presentation_score.round(2).tolist()
[0.1, 0.96, 0.9, 0.95]
```

## Scanning protein sequences

The {meth}`~mhcflurry.Class1PresentationPredictor.predict_sequences` method supports
scanning protein sequences for MHC ligands. Here's an example to identify all
peptides with a predicted binding affinity of 500 nM or tighter to any allele
across two sample genotypes and two short peptide sequences.

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
>>> len(scan)
10
>>> bool(scan.affinity.le(500).all())
True
>>> scan[["sequence_name", "peptide", "best_allele"]].head(2).to_dict("records")
[{'sequence_name': 'protein1', 'peptide': 'LLLVVSNLL', 'best_allele': 'A0201'}, {'sequence_name': 'protein1', 'peptide': 'LLLLVVSNL', 'best_allele': 'A0201'}]
```

When using `predict_sequences`, the flanking sequences for each peptide are
automatically included in the processing and presentation predictions.

See the documentation for {class}`~mhcflurry.Class1PresentationPredictor` for other
useful methods.


## Lower level interfaces

The {class}`~mhcflurry.Class1PresentationPredictor` delegates to a
{class}`~mhcflurry.Class1AffinityPredictor` instance for binding affinity predictions.
If all you need are binding affinities, you can use this instance directly.

Here's an example:

```{doctest}
>>> from mhcflurry import Class1AffinityPredictor
>>> predictor = Class1AffinityPredictor.load()
>>> affinities = predictor.predict_to_dataframe(
...     allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])
>>> affinities[["peptide", "allele"]].to_dict("records")
[{'peptide': 'SIINFEKL', 'allele': 'HLA-A*02:01'}, {'peptide': 'SIINFEQL', 'allele': 'HLA-A*02:01'}]
>>> bool(((affinities.prediction_low < affinities.prediction) &
...       (affinities.prediction < affinities.prediction_high)).all())
True
```

The `prediction_low` and `prediction_high` fields give the 5-95 percentile
predictions across the models in the ensemble. This detailed information is not
available through the higher-level {class}`~mhcflurry.Class1PresentationPredictor`
interface.

Under the hood, `Class1AffinityPredictor` itself delegates to an ensemble of
of {class}`~mhcflurry.Class1NeuralNetwork` instances, which implement the neural network
models used for prediction. To fit your own affinity prediction models, call
{meth}`~mhcflurry.Class1NeuralNetwork.fit`.

You can similarly use {class}`~mhcflurry.Class1ProcessingPredictor` directly for
antigen processing prediction, and there is a low-level
{class}`~mhcflurry.Class1ProcessingNeuralNetwork` with a {meth}`~mhcflurry.Class1ProcessingNeuralNetwork.fit` method.

See the API documentation of these classes for details.
