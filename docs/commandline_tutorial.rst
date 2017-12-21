Command-line usage
==================

Downloading models
------------------

Most users will use pre-trained MHCflurry models that we release. These models
are distributed separately from the source code and may be downloaded with the
following command:

.. code: shell
    $ mhcflurry-downloads fetch models_class1

We also release other "downloads," such as curated training data and some
experimental models. To see what you have downloaded, run:

.. code: shell
    $ mhcflurry-downloads info


mhcflurry-predict
-----------------

The ``mhcflurry-predict`` command generates predictions from the command-line.
It defaults to using the pre-trained models you downloaded above but this can
be customized with the ``--models`` argument. See ``mhcflurry-predict -h`` for
details.

.. command-output:: mhcflurry-predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD SIINFEKQ
    :nostderr:

The predictions returned are affinities (KD) in nM. The ``prediction_low`` and
``prediction_high`` fields give the 5-95 percentile predictions across
the models in the ensemble. The predictions above were generated with MHCflurry
|version|.

Your exact predictions may vary slightly from these (up to about 1 nM) depending
on the Keras backend in use and other numerical details. Different versions of
MHCflurry can of course give results considerably different from these.

You can also specify the input and output as CSV files. Run
``mhcflurry-predict -h`` for details.

Fitting your own models
-----------------------

Scanning protein sequences for predicted epitopes
-------------------------------------------------

The `mhctools <https://github.com/hammerlab/mhctools>`__ package
provides support for scanning protein sequences to find predicted
epitopes. It supports MHCflurry as well as other binding predictors.
Here is an example.

First, install ``mhctools`` if it is not already installed:

.. code:: shell

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
        --output-csv /tmp/result.csv
    :ellipsis: 2,-2
    :nostderr:

This will write a file giving predictions for all subsequences of the specified lengths:

.. command-output::
    head -n 3 /tmp/result.csv
