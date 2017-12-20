Details on the released models
===============================

The released MHCflurry predictor consists of an ensemble of eight models for each
supported allele. Each model in the ensemble was trained on a random 80% sample
of the data for the allele, and the remaining 20% was used for early stopping.
All models use the same architecture. The predictions are taken to be the geometric
mean of the nM binding affinity predictions of the individual models. The script
we run to train these models is in "downloads-generation/models_class1/GENERATE.sh"
in the repository.

Neural network architecture
-------------------------------------------------------------

The neural network architecture is quite simple, consisting of a locally
connected layer, a dense layer, and a sigmoid output.

.. include:: _build/_models_info.rst

Architecture diagram:

.. image:: _build/_models_architecture.png


Cross validation performance
-------------------------------------------------------------

The accuracy of the MHCflurry downloadable models was estimated using 5-fold cross
validation on the training data. The values shown here are the mean cross validation
scores across folds.

The AUC and F1 estimates use a 500 nM cutoff for distinguishing strong-binders
from weak- or non-binders. The Kendall Tau score gives the rank correlation
between the predicted and measured affinities; it uses no cutoff.

.. include:: _build/_models_cv.rst
