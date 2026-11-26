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

    $ mhcflurry-downloads fetch models_class1

Files downloaded with :ref:`mhcflurry-downloads` are stored in a platform-specific
directory. To get the path to downloaded data, you can use:

.. command-output:: mhcflurry-downloads path models_class1
    :nostderr:

We also release a few other "downloads," such as curated training data and some
experimental models. To see what's available and what you have downloaded, run:

.. command-output:: mhcflurry-downloads info
    :nostderr:

.. note::

    The code we use for *generating* the downloads is in the
    ``downloads_generation`` directory in the repository.


Generating predictions
----------------------

The :ref:`mhcflurry-predict` command generates predictions from the command-line.
By default it will use the pre-trained models you downloaded above; other
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

The predictions are given as affinities (KD) in nM in the ``mhcflurry_prediction``
column. The other fields give the 5-95 percentile predictions across
the models in the ensemble and the quantile of the affinity prediction among
a large number of random peptides tested on that allele.

The predictions shown above were generated with MHCflurry |version|. Different versions of
MHCflurry can give considerably different results. Even
on the same version, exact predictions may vary (up to about 1 nM) depending
on the Keras backend and other details.

In most cases you'll want to specify the input as a CSV file instead of passing
peptides and alleles as commandline arguments. See :ref:`mhcflurry-predict` docs.

Fitting your own models
-----------------------

The :ref:`mhcflurry-class1-train-allele-specific-models` command is used to
fit models to training data. The models we release with MHCflurry are trained
with a command like:

.. code-block:: shell

    $ mhcflurry-class1-train-allele-specific-models \
        --data TRAINING_DATA.csv \
        --hyperparameters hyperparameters.yaml \
        --min-measurements-per-allele 75 \
        --out-models-dir models

MHCflurry predictors are serialized to disk as many files in a directory. The
command above will write the models to the output directory specified by the
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

To call :ref:`mhcflurry-class1-train-allele-specific-models` you'll need some
training data. The data we use for our released predictors can be downloaded with
:ref:`mhcflurry-downloads`:

.. code-block:: shell

    $ mhcflurry-downloads fetch data_curated

It looks like this:

.. command-output::
    bzcat "$(mhcflurry-downloads path data_curated)/curated_training_data.no_mass_spec.csv.bz2" | head -n 3
    :shell:
    :nostderr:


Scanning protein sequences for predicted epitopes
-------------------------------------------------

The `mhctools <https://github.com/hammerlab/mhctools>`__ package
provides support for scanning protein sequences to find predicted
epitopes. It supports MHCflurry as well as other binding predictors.
Here is an example.

First, install ``mhctools`` if it is not already installed:

.. code-block:: shell

    $ pip install mhctools

We'll generate predictions across ``example.fasta``, a FASTA file with two short
sequences:

.. literalinclude:: /example.fasta

Here's the ``mhctools`` invocation. See ``mhctools -h`` for more information.

.. command-output::
    mhctools
        --mhc-predictor mhcflurry
        --input-fasta-file example.fasta
        --mhc-alleles A02:01,A03:01
        --mhc-peptide-lengths 8,9,10,11
        --extract-subsequences
        --output-csv /tmp/subsequence_predictions.csv
    :ellipsis: 2,-2
    :nostderr:

This will write a file giving predictions for all subsequences of the specified lengths:

.. command-output::
    head -n 3 /tmp/subsequence_predictions.csv
