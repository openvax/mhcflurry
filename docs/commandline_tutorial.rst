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

