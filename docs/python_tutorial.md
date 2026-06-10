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
>>> predictor.predict(
...     peptides=["SIINFEKL", "NLVPMVATV"],
...     alleles=["HLA-A0201", "HLA-A0301"],
...     verbose=0)
     peptide  peptide_num sample_name      affinity best_allele  processing_score  presentation_score  presentation_percentile
0   SIINFEKL            0     sample1  11927.173394   HLA-A0201          0.264710            0.020690                11.630326
1  NLVPMVATV            1     sample1     16.570969   HLA-A0201          0.533008            0.970187                 0.018723
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
>>> predictor.predict(
...     peptides=["KSEYMTSWFY", "NLVPMVATV"],
...     alleles={
...        "sample1": ["A0201", "A0301", "B0702", "B4402", "C0201", "C0702"],
...        "sample2": ["A0101", "A0206", "B5701", "C0202"],
...     },
...     verbose=0)
      peptide  peptide_num sample_name     affinity best_allele  processing_score  presentation_score  presentation_percentile
0  KSEYMTSWFY            0     sample1  8292.186793       C0201          0.542474            0.074376                 3.639185
1   NLVPMVATV            1     sample1    16.570969       A0201          0.533008            0.970187                 0.018723
2  KSEYMTSWFY            0     sample2    88.898412       A0101          0.542474            0.868106                 0.171848
3   NLVPMVATV            1     sample2    17.406640       A0206          0.533008            0.968773                 0.021413
```

Here the strongest binder for each sample / peptide pair is returned.

Many users will focus on the binding affinity predictions, as the
processing and presentation predictions are experimental. If you do use the latter
scores, however, when available you should provide the upstream (N-flank)
and downstream (C-flank) sequences from the source proteins of the peptides for
a small boost in accuracy. To do so, specify the `n_flank` and `c_flank`
arguments, which give the flanking sequences for the corresponding peptides:

```{doctest}
>>> predictor.predict(
...     peptides=["KSEYMTSWFY", "NLVPMVATV"],
...     n_flanks=["NNNNNNN", "SSSSSSSS"],
...     c_flanks=["CCCCCCCC", "YYYAAAA"],
...     alleles={
...        "sample1": ["A0201", "A0301", "B0702", "B4402", "C0201", "C0702"],
...        "sample2": ["A0101", "A0206", "B5701", "C0202"],
...     },
...     verbose=0)
      peptide   n_flank   c_flank  peptide_num sample_name     affinity best_allele  processing_score  presentation_score  presentation_percentile
0  KSEYMTSWFY   NNNNNNN  CCCCCCCC            0     sample1  8292.186793       C0201          0.626173            0.097041                 2.991685
1   NLVPMVATV  SSSSSSSS   YYYAAAA            1     sample1    16.570969       A0201          0.436871            0.956961                 0.036957
2  KSEYMTSWFY   NNNNNNN  CCCCCCCC            0     sample2    88.898412       A0101          0.626173            0.898967                 0.122663
3   NLVPMVATV  SSSSSSSS   YYYAAAA            1     sample2    17.406640       A0206          0.436871            0.954945                 0.041168
```

## Scanning protein sequences

The {meth}`~mhcflurry.Class1PresentationPredictor.predict_sequences` method supports
scanning protein sequences for MHC ligands. Here's an example to identify all
peptides with a predicted binding affinity of 500 nM or tighter to any allele
across two sample genotypes and two short peptide sequences.

```{doctest}
>>> predictor.predict_sequences(
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
  sequence_name  pos     peptide n_flank c_flank sample_name    affinity best_allele  affinity_percentile  processing_score  presentation_score  presentation_percentile
0      protein1   14   LLLVVSNLL   GSRLL             sample1   57.180447       A0201             0.398125          0.233159            0.754186                 0.351359
1      protein1   13   LLLLVVSNL   KGSRL       L     sample1   57.338895       A0201             0.398125          0.030920            0.586465                 0.642908
2      protein1    5   SSQKGSRLL   MDSKG   LLLVV     sample2  110.778519       C0202             0.781875          0.060995            0.455738                 0.920299
3      protein1    6   SQKGSRLLL   DSKGS   LLVVS     sample2  254.479925       C0202             1.734875          0.101657            0.303070                 1.356196
4      protein1   13  LLLLVVSNLL   KGSRL             sample1  260.390251       A0201             1.012500          0.158010            0.345066                 1.214701
5      protein1   12  LLLLLVVSNL   QKGSR       L     sample1  308.149631       A0201             1.093750          0.015113            0.206176                 1.801603
6      protein2    0   SSLPTPEDK           EQAQQ     sample2  410.354088       C0202             2.398000          0.003090            0.158057                 2.154946
7      protein1    5    SSQKGSRL   MDSKG   LLLLV     sample2  444.320680       C0202             2.511750          0.026198            0.159450                 2.137962
8      protein2    0   SSLPTPEDK           EQAQQ     sample1  459.295763       A0301             0.970625          0.003090            0.143999                 2.292011
9      protein1    4   GSSQKGSRL    MDSK   LLLLV     sample2  469.052760       C0202             2.594750          0.013744            0.146487                 2.261060
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
>>> predictor.predict_to_dataframe(allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])
    peptide       allele    prediction  prediction_low  prediction_high  prediction_percentile
0  SIINFEKL  HLA-A*02:01  11927.160672     6901.075753     18127.365636               6.296000
1  SIINFEQL  HLA-A*02:01  12070.888207     7362.362111     18465.971144               6.354125
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
