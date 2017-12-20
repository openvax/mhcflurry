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

.. code:: shell

    $ mhcflurry-predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD SIINFEKQ
    allele,peptide,mhcflurry_prediction,mhcflurry_prediction_low,mhcflurry_prediction_high
    HLA-A0201,SIINFEKL,5326.541919062165,3757.86675352994,7461.37693353508
    HLA-A0201,SIINFEKD,18763.70298522213,13140.82000240037,23269.82139560844
    HLA-A0201,SIINFEKQ,18620.10057358322,13096.425874678192,23223.148184869413
    HLA-A0301,SIINFEKL,24481.726678691946,21035.52779725433,27245.371837497867
    HLA-A0301,SIINFEKD,24687.529360239587,21582.590014592537,27749.39869616437
    HLA-A0301,SIINFEKQ,25923.062203902562,23522.5793450799,28079.456657427705

The predictions returned are affinities (KD) in nM. The ``prediction_low`` and
``prediction_high`` fields give the 5-95 percentile predictions across
the models in the ensemble. The predictions above were generated with MHCflurry
0.9.2.

Your exact predictions may vary slightly from these (up to about 1 nM) depending
on the Keras backend in use and other numerical details. Different versions of
MHCflurry can of course give results considerably different from these.

You can also specify the input and output as CSV files. Run
``mhcflurry-predict -h`` for details.

Fitting your own models
-----------------------

