Introduction and setup
=======================

MHCflurry is an open source package for peptide/MHC I binding affinity prediction. It
provides competitive accuracy with a fast and documented implementation.

You can download pre-trained MHCflurry models fit to affinity measurements
deposited in IEDB or train a MHCflurry predictor on your own data.

Currently only allele-specific prediction is implemented, in which separate models
are trained for each allele. The released models therefore support a fixed set of common
class I alleles for which sufficient published training data is available
(see :ref:`models_supported_alleles`\ ).

MHCflurry supports Python versions 2.7 and 3.4+. It uses the `keras <https://keras.io>`__
neural network library via either the Tensorflow or Theano backends. GPUs may
optionally be used for a generally modest speed improvement.

If you find MHCflurry useful in your research please cite:

    O'Donnell, T. et al., 2017. MHCflurry: open-source class I MHC
    binding affinity prediction. bioRxiv. Available at:
    http://www.biorxiv.org/content/early/2017/08/09/174243.


Installation (pip)
-------------------

Install the package:

.. code-block:: shell

    $ pip install mhcflurry

Then download our datasets and trained models:

.. code-block:: shell

    $ mhcflurry-downloads fetch

From a checkout you can run the unit tests with:

.. code-block:: shell

    $ pip install nose
    $ nosetests .


Using conda
-------------

You can alternatively get up and running with a `conda <https://conda.io/docs/>`__
environment as follows. Some users have reported that this can avoid problems installing
tensorflow.

.. code-block:: shell

    $ conda create -q -n mhcflurry-env python=3.6 'tensorflow>=1.1.2'
    $ source activate mhcflurry-env

Then continue as above:

.. code-block:: shell

    $ pip install mhcflurry
    $ mhcflurry-downloads fetch

