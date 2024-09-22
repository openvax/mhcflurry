Introduction and setup
=======================

MHCflurry is an open source package for peptide/MHC I binding affinity prediction. It
aims to provide competitive accuracy with a fast and documented implementation.

You can download pre-trained MHCflurry models fit to mass spec-identified MHC I
ligands and peptide/MHC affinity measurements deposited in IEDB (plus a few other
sources) or train a MHCflurry predictor on your own data.

Starting in version 1.6.0, the default MHCflurry binding affinity predictors
are "pan-allele" models that support most sequenced MHC I alleles across humans
and a few other species (about 14,000 alleles in total). This version also
introduces two experimental predictors, an "antigen processing" predictor
that attempts to model MHC allele-independent effects such as proteosomal
cleavage and a "presentation" predictor that integrates processing predictions
with binding affinity predictions to give a composite "presentation score." Both
models are trained on mass spec-identified MHC ligands.

MHCflurry supports Python 3.4+. It uses the `keras <https://keras.io>`__
neural network library via either the Tensorflow or Theano backends. GPUs may
optionally be used for a modest speed improvement.

If you find MHCflurry useful in your research, please cite:

    T. J. O'Donnell, et al. "MHCflurry 2.0: Improved pan-allele prediction of MHC
    I-presented peptides by incorporating antigen processing,"
    *Cell Systems*, 2020. https://doi.org/10.1016/j.cels.2020.06.010

    T. J. O’Donnell, et al., "MHCflurry: Open-Source Class I MHC Binding Affinity
    Prediction," *Cell Systems*, 2018. https://doi.org/10.1016/j.cels.2018.05.014

If you have questions or encounter problems, please file an issue at the
MHCflurry github repo: https://github.com/openvax/mhcflurry


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

    $ pip install pytest
    $ pytest


Using conda
-------------

You can alternatively get up and running with a `conda <https://conda.io/docs/>`__
environment as follows. Some users have reported that this can avoid problems installing
tensorflow.

.. code-block:: shell

    $ conda create -q -n mhcflurry-env python=3.8 tensorflow
    $ source activate mhcflurry-env

Then continue as above:

.. code-block:: shell

    $ pip install mhcflurry
    $ mhcflurry-downloads fetch

