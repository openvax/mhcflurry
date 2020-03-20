Python library tutorial
=======================

The MHCflurry Python API exposes additional options and features beyond those
supported by the commandline tools and can be more convenient for interactive
analyses and bioinformatic pipelines. This tutorial gives a basic overview
of the most important functionality. See the :ref:`API-documentation` for further
details.

Loading a predictor
----------------------------------

Most prediction tasks can be performed using the
`~mhcflurry.Class1PresentationPredictor` class, which provides a programmatic API
to the functionality in the :ref:`mhcflurry-predict` and
:ref:`mhcflurry-predict-scan` commands.

Instances of `~mhcflurry.Class1PresentationPredictor` wrap a
`~mhcflurry.Class1AffinityPredictor` to generate binding affinity predictions
and a `~mhcflurry.Class1ProcessingPredictor` to generate antigen processing
predictions. The presentation score is computed using a logistic regression
model over binding affinity and processing predictions.

Use the `~mhcflurry.Class1PresentationPredictor.load` static method to load a
trained predictor from disk. With no arguments this method will load the predictor
released with MHCflurry (see :ref:`downloading`\ ). If you pass a path to a
models directory, then it will load that predictor instead.

.. doctest::

    >>> from mhcflurry import Class1PresentationPredictor
    >>> predictor = Class1PresentationPredictor.load()
    >>> predictor.supported_alleles[:5]
    ['Atbe-B*01:01', 'Atbe-E*03:01', 'Atbe-G*03:01', 'Atbe-G*03:02', 'Atbe-G*06:01']

Predicting for individual peptides
----------------------------------

To generate predictions for individual peptides, we can use the
`~mhcflurry.Class1AffinityPredictor.predict` method of the `~mhcflurry.Class1PresentationPredictor`,
loaded above. This method returns a `pandas.DataFrame` with binding affinity, processing, and presentation
predictions:

.. doctest::

    >>> predictor.predict(
    ...     peptides=["SIINFEKL", "NLVPMVATV"],
    ...     alleles=["HLA-A0201", "HLA-A0301"],
    ...     verbose=0)
         peptide  peptide_num sample_name      affinity best_allele  processing_score  presentation_score
    0   SIINFEKL            0     sample1  12906.786173   HLA-A0201          0.101473            0.012503
    1  NLVPMVATV            1     sample1     15.038358   HLA-A0201          0.676289            0.975463

Here, the list of alleles is taken to be an individual's MHC I genotype (i.e. up
to 6 alleles), and the strongest binder across alleles for each peptide is
reported.

.. note::

    MHCflurry normalizes allele names using the `mhcnames <https://github.com/openvax/mhcnames>`__
    package. Names like ``HLA-A0201`` or ``A*02:01`` will be
    normalized to ``HLA-A*02:01``, so most naming conventions can be used
    with methods such as `~mhcflurry.Class1PresentationPredictor.predict`.

If you have multiple sample genotypes, you can pass a dict, where the
keys are arbitrary sample names:

.. doctest::

    >>> predictor.predict(
    ...     peptides=["KSEYMTSWFY", "NLVPMVATV"],
    ...     alleles={
    ...        "sample1": ["A0201", "A0301", "B0702", "B4402", "C0201", "C0702"],
    ...        "sample2": ["A0101", "A0206", "B5701", "C0202"],
    ...     },
    ...     verbose=0)
          peptide  peptide_num sample_name      affinity best_allele  processing_score  presentation_score
    0  KSEYMTSWFY            0     sample1  16737.745268       A0301          0.381632            0.026550
    1   NLVPMVATV            1     sample1     15.038358       A0201          0.676289            0.975463
    2  KSEYMTSWFY            0     sample2     62.540779       A0101          0.381632            0.796731
    3   NLVPMVATV            1     sample2     15.765500       A0206          0.676289            0.974439

Here the strongest binder for each sample / peptide pair is returned.

Many users will focus on the binding affinity predictions, as the
processing and presentation predictions are experimental. If you do use the latter
scores, however, when available you should provide the upstream (N-flank)
and downstream (C-flank) sequences from the source proteins of the peptides for
a small boost in accuracy. To do so, specify the ``n_flank`` and ``c_flank``
arguments, which give the flanking sequences for the corresponding peptides:

.. doctest::

    >>> predictor.predict(
    ...     peptides=["KSEYMTSWFY", "NLVPMVATV"],
    ...     n_flanks=["NNNNNNN", "SSSSSSSS"],
    ...     c_flanks=["CCCCCCCC", "YYYAAAA"],
    ...     alleles={
    ...        "sample1": ["A0201", "A0301", "B0702", "B4402", "C0201", "C0702"],
    ...        "sample2": ["A0101", "A0206", "B5701", "C0202"],
    ...     },
    ...     verbose=0)
          peptide   n_flank   c_flank  peptide_num sample_name      affinity best_allele  processing_score  presentation_score
    0  KSEYMTSWFY   NNNNNNN  CCCCCCCC            0     sample1  16737.745268       A0301          0.605816            0.056190
    1   NLVPMVATV  SSSSSSSS   YYYAAAA            1     sample1     15.038358       A0201          0.824994            0.986719
    2  KSEYMTSWFY   NNNNNNN  CCCCCCCC            0     sample2     62.540779       A0101          0.605816            0.897493
    3   NLVPMVATV  SSSSSSSS   YYYAAAA            1     sample2     15.765500       A0206          0.824994            0.986155

Scanning protein sequences
--------------------------

The `~mhcflurry.Class1PresentationPredictor.predict_sequences` method supports
scanning protein sequences for MHC ligands. Here's an example to identify all
peptides with a predicted binding affinity of 500 nM or tighter to any allele
across two sample genotypes and two short peptide sequences.

.. doctest::

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
      sequence_name  pos     peptide         n_flank     c_flank sample_name    affinity best_allele  affinity_percentile  processing_score  presentation_score
    0      protein1   13   LLLLVVSNL   MDSKGSSQKGSRL           L     sample1   38.206225       A0201             0.380125          0.017644            0.571060
    1      protein1   14   LLLVVSNLL  MDSKGSSQKGSRLL                 sample1   42.243472       A0201             0.420250          0.090984            0.619213
    2      protein1    5   SSQKGSRLL           MDSKG   LLLVVSNLL     sample2   66.749223       C0202             0.803375          0.383608            0.774468
    3      protein1    6   SQKGSRLLL          MDSKGS    LLVVSNLL     sample2  178.033467       C0202             1.820000          0.275019            0.482206
    4      protein1   13  LLLLVVSNLL   MDSKGSSQKGSRL                 sample1  202.208167       A0201             1.112500          0.058782            0.261320
    5      protein1   12  LLLLLVVSNL    MDSKGSSQKGSR           L     sample1  202.506582       A0201             1.112500          0.010025            0.225648
    6      protein2    0   SSLPTPEDK                    EQAQQTHH     sample1  335.529377       A0301             1.011750          0.010443            0.156798
    7      protein2    0   SSLPTPEDK                    EQAQQTHH     sample2  353.451759       C0202             2.674250          0.010443            0.150753
    8      protein1    8   KGSRLLLLL        MDSKGSSQ      VVSNLL     sample2  410.327286       C0202             2.887000          0.121374            0.194081
    9      protein1    5    SSQKGSRL           MDSKG  LLLLVVSNLL     sample2  477.285937       C0202             3.107375          0.111982            0.168572

When using ``predict_sequences``, the flanking sequences for each peptide are
automatically included in the processing and presentation predictions.

See the documentation for `~mhcflurry.Class1PresentationPredictor` for other
useful methods.


Lower level interfaces
----------------------------------

The `~mhcflurry.Class1PresentationPredictor` predictor delegates to a
`~mhcflurry.Class1AffinityPredictor` instance for binding affinity predictions.
If all you need are binding affinities, you can use this instance directly.

Here's an example:

.. doctest::

    >>> from mhcflurry import Class1AffinityPredictor
    >>> predictor = Class1AffinityPredictor.load()
    >>> predictor.predict_to_dataframe(allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])
        peptide     allele    prediction  prediction_low  prediction_high  prediction_percentile
    0  SIINFEKL  HLA-A0201  12906.786173     8829.460289     18029.923061               6.566375
    1  SIINFEQL  HLA-A0201  13025.300796     9050.056312     18338.004869               6.623625

The ``prediction_low`` and ``prediction_high`` fields give the 5-95 percentile
predictions across the models in the ensemble. This detailed information is not
available through the higher-level `~mhcflurry.Class1PresentationPredictor`
interface.

Under the hood, `Class1AffinityPredictor` itself delegates to an ensemble of
of `~mhcflurry.Class1NeuralNetwork` instances, which implement the neural network
models used for prediction. To fit your own affinity prediction models, call
`~mhcflurry.Class1NeuralNetwork.fit`.

You can similarly use `~mhcflurry.Class1ProcessingPredictor` directly for
antigen processing prediction, and there is a low-level
`~mhcflurry.Class1ProcessingNeuralNetwork` with a `~mhcflurry.Class1ProcessingNeuralNetwork.fit` method.

See the API documentation of these classes for details.