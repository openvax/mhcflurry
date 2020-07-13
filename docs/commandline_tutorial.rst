.. _commandline_tutorial:

Command-line tutorial
=====================

.. _downloading:

Downloading models
------------------

Most users will use pre-trained MHCflurry models that we release. These models
are distributed separately from the pip package and may be downloaded with the
:ref:`mhcflurry-downloads` tool:

.. code-block:: shell

    $ mhcflurry-downloads fetch models_class1_presentation

Files downloaded with :ref:`mhcflurry-downloads` are stored in a platform-specific
directory. To get the path to downloaded data, you can use:

.. command-output:: mhcflurry-downloads path models_class1_presentation
    :nostderr:

We also release a number of other "downloads," such as curated training data and some
experimental models. To see what's available and what you have downloaded, run
``mhcflurry-downloads info``.

Most users will only need ``models_class1_presentation``, however, as the
presentation predictor includes a peptide / MHC I binding affinity (BA) predictor
as well as an antigen processing (AP) predictor.

.. note::

    The code we use for *generating* the downloads is in the
    ``downloads_generation`` directory in the repository (https://github.com/openvax/mhcflurry/tree/master/downloads-generation)


Generating predictions
----------------------

The :ref:`mhcflurry-predict` command generates predictions for individual peptides
(see the next section for how to scan protein sequences for epitopes). By
default it will use the pre-trained models you downloaded above. Other
models can be used by specifying the ``--models`` argument.

Running:

.. command-output::
    mhcflurry-predict
        --alleles HLA-A0201 HLA-A0301
        --peptides SIINFEKL SIINFEKD SIINFEKQ
        --out /tmp/predictions.csv
    :nostderr:

results in a file like this:

.. command-output::
    cat /tmp/predictions.csv

The binding affinity predictions are given as affinities (KD) in nM in the
``mhcflurry_affinity`` column. Lower values indicate stronger binders. A commonly-used
threshold for peptides with a reasonable chance of being immunogenic is 500 nM.

The ``mhcflurry_affinity_percentile`` gives the percentile of the affinity
prediction among a large number of random peptides tested on that allele (range
0 - 100). Lower is stronger. Two percent is a commonly-used threshold.

The last two columns give the antigen processing and presentation scores,
respectively. These range from 0 to 1 with higher values indicating more
favorable processing or presentation.

.. note::

    The processing predictor is experimental. It models allele-independent
    effects that influence whether a
    peptide will be detected in a mass spec experiment. The presentation score is
    a simple logistic regression model that combines the (log) binding affinity
    prediction with the processing score to give a composite prediction. The resulting
    prediction may be useful for prioritizing potential epitopes, but no
    thresholds have been established for what constitutes a "high enough"
    presentation score.

In most cases you'll want to specify the input as a CSV file instead of passing
peptides and alleles as commandline arguments. If you're relying on the
processing or presentation scores, you may also want to pass the upstream and
downstream sequences of the peptides from their source proteins for potentially more
accurate cleavage prediction. See the :ref:`mhcflurry-predict` docs.


Using the older, allele-specific models
-------------------------------------------

Previous versions of MHCflurry (described in the 2018 paper) used models
trained on affinity measurements, one allele per model (i.e. allele-specific).
Mass spec datasets were incorporated in the model selection step.

These models are still available to use with the latest version of MHCflurry.
To download these predictors, run:

.. code-block:: shell

    $ mhcflurry-downloads fetch models_class1

and specify ``--models`` when you call ``mhcflurry-predict``:


.. code-block:: shell

    $ mhcflurry-predict \
        --alleles HLA-A0201 HLA-A0301 \
        --peptides SIINFEKL SIINFEKD SIINFEKQ \
        --models "$(mhcflurry-downloads path models_class1)/models"
        --out /tmp/predictions.csv


Scanning protein sequences for predicted MHC I ligands
-------------------------------------------------

Starting in version 1.6.0, MHCflurry supports scanning proteins for MHC-binding
peptides using the ``mhcflurry-predict-scan`` command.

We'll generate predictions across ``example.fasta``, a FASTA file with two short
sequences:

.. literalinclude:: /example.fasta

Here's the ``mhcflurry-predict-scan`` invocation to scan the proteins for
binders to either of two MHC I genotypes (using a 100 nM threshold):

.. command-output::
    mhcflurry-predict-scan
        example.fasta
        --alleles
            HLA-A*02:01,HLA-A*03:01,HLA-B*57:01,HLA-B*45:01,HLA-C*02:02,HLA-C*07:02
            HLA-A*01:01,HLA-A*02:06,HLA-B*44:02,HLA-B*07:02,HLA-C*01:02,HLA-C*03:01
        --results-filtered affinity
        --threshold-affinity 100
    :nostderr:

See the :ref:`mhcflurry-predict-scan` docs for more options.


Fitting your own models
-----------------------

If you have your own data and want to fit your own MHCflurry models, you have
a few options. If you have data for only one or a few MHC I alleles, the best
approach is to use the
:ref:`mhcflurry-class1-train-allele-specific-models` command to fit an
"allele-specific" predictor, in which separate neural networks are used for
each allele.

To call :ref:`mhcflurry-class1-train-allele-specific-models` you'll need some
training data. The data we use for our released predictors can be downloaded with
:ref:`mhcflurry-downloads`:

.. code-block:: shell

    $ mhcflurry-downloads fetch data_curated

It looks like this:

.. command-output::
    bzcat "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" | head -n 3
    :shell:
    :nostderr:

Here's an example invocation to fit a predictor:

.. code-block:: shell

    $ mhcflurry-class1-train-allele-specific-models \
        --data curated_training_data.csv.bz2 \
        --hyperparameters hyperparameters.yaml \
        --min-measurements-per-allele 75 \
        --out-models-dir models

The ``hyperparameters.yaml`` file gives the list of neural network architectures
to train models for. Here's an example specifying a single architecture:

.. code-block:: yaml

    - activation: tanh
      dense_layer_l1_regularization: 0.0
      dropout_probability: 0.0
      early_stopping: true
      layer_sizes: [8]
      locally_connected_layers: []
      loss: custom:mse_with_inequalities
      max_epochs: 500
      minibatch_size: 128
      n_models: 4
      output_activation: sigmoid
      patience: 20
      peptide_amino_acid_encoding: BLOSUM62
      random_negative_affinity_max: 50000.0
      random_negative_affinity_min: 20000.0
      random_negative_constant: 25
      random_negative_rate: 0.0
      validation_split: 0.1

The available hyperparameters for binding predictors are defined in
`~mhcflurry.Class1NeuralNetwork`. To see exactly how
these are used you will need to read the source code.

.. note::

    MHCflurry predictors are serialized to disk as many files in a directory. The
    model training command above will write the models to the output directory specified by the
    ``--out-models-dir`` argument. This directory has files like:

    .. program-output::
        ls "$(mhcflurry-downloads path models_class1)/models"
        :shell:
        :nostderr:
        :ellipsis: 4,-4

    The ``manifest.csv`` file gives metadata for all the models used in the predictor.
    There will be a ``weights_...`` file for each model giving its weights
    (the parameters for the neural network). The ``percent_ranks.csv`` stores a
    histogram of model predictions for each allele over a large number of random
    peptides. It is used for generating the percent ranks at prediction time.

To fit pan-allele models like the ones released with MHCflurry, you can use
a similar tool, :ref:`mhcflurry-class1-train-pan-allele-models`. You'll probably
also want to take a look at the scripts used to generate the production models,
which are available in the *downloads-generation* directory in the MHCflurry
repository. See the scripts in the *models_class1_pan* subdirectory to see how the
fitting and model selection was done for models currently distributed with MHCflurry.

.. note::

    The production MHCflurry models were fit using a cluster with several
    dozen GPUs over a period of about two days. If you model select over fewer
    architectures, however, it should be possible to fit a predictor using less
    resources.


Environment variables
-------------------------------------------------

MHCflurry behavior can be modified using these environment variables:

``MHCFLURRY_DEFAULT_CLASS1_MODELS``
    Path to models directory. If you call ``Class1AffinityPredictor.load()``
    with no arguments, the models specified in this environment variable will be
    used. If this environment variable is undefined, the downloaded models for
    the current MHCflurry release are used.

``MHCFLURRY_OPTIMIZATION_LEVEL``
    The pan-allele models can be somewhat slow. As an optimization, when this
    variable is greater than 0 (default is 1), we "stitch" the pan-allele models in
    the ensemble into one large tensorflow graph. In our experiments
    it gives about a 30% speed improvement. It has no effect on allele-specific
    models. Set this variable to 0 to disable this behavior. This may be helpful
    if you are running out of memory using the pan-allele models.


``MHCFLURRY_DEFAULT_PREDICT_BATCH_SIZE``
    For large prediction tasks, it can be helpful to increase the prediction batch
    size, which is set by this environment variable (default is 4096). This
    affects both allele-specific and pan-allele predictors. It can have large
    effects on performance. Alternatively, if you are running out of memory,
    you can try decreasing the batch size.


