Details on the released predictor
==================================

The released MHCflurry predictor consists of an ensemble of 8-16 models for each
supported allele. Each model in the ensemble was trained on a random 80% sample
of the data for the allele, with 10% held out for early stopping and 10%
held out for model selection. Model selection additionally made use of mass-spec
data when available for an allele.

The predictions are taken to be the geometric mean of the nM binding affinity
predictions of the individual models. The script we run to train these models is in
"downloads-generation/models_class1/GENERATE.sh" in the repository.


Alleles
-------------------------------------------------------------

.. include:: /_build/_alleles_info.rst


Neural network architectures
-------------------------------------------------------------

.. include:: /_build/_models_info.rst

