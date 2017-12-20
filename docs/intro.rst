Introduction and setup
=======================

MHCflurry is a Python package for peptide/MHC I binding affinity prediction. It
provides competitive accuracy with a fast, documented, open source
implementation.

You can download pre-trained MHCflurry models fit to affinity measurements
deposited in IEDB. See the "downloads_generation/models_class1" directory in the
repository for the workflow used to train these predictors. Users with their own
data can also fit their own MHCflurry models.

Currently only allele-specific prediction is implemented, in which separate models
are trained for each allele. The released models therefore support a fixed set of common
class I alleles for which sufficient published training data is available.

MHCflurry supports Python versions 2.7 and 3.4+. It uses the Keras neural
network library via either the Tensorflow or Theano backends. GPUs may
optionally be used for a generally modest speed improvement.

If you find MHCflurry useful in your research please cite:

    O'Donnell, T. et al., 2017. MHCflurry: open-source class I MHC
    binding affinity prediction. bioRxiv. Available at:
    http://www.biorxiv.org/content/early/2017/08/09/174243.


Installation (pip)
-------------------

Install the package:

::

    pip install mhcflurry

Then download our datasets and trained models:

::

    mhcflurry-downloads fetch

From a checkout you can run the unit tests with:

::

    pip install nose
    nosetests .


Using conda
-------------

You can alternatively get up and running with a `conda <https://conda.io/docs/>`__
environment as follows. Some users have reported that this can avoid problems installing
tensorflow.

::

    conda create -q -n mhcflurry-env python=3.6 'tensorflow>=1.1.2'
    source activate mhcflurry-env

Then continue as above:

::

    pip install mhcflurry
    mhcflurry-downloads fetch

