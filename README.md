[![Build Status](https://travis-ci.org/openvax/mhcflurry.svg?branch=master)](https://travis-ci.org/openvax/mhcflurry)

# mhcflurry
[MHC I](https://en.wikipedia.org/wiki/MHC_class_I) ligand
prediction package with competitive accuracy and a fast and 
[documented](http://openvax.github.io/mhcflurry/) implementation.

MHCflurry supports Class I peptide/MHC binding affinity prediction using
ensembles of allele-specific models. It runs on Python 2.7 and 3.4+ using
the [keras](https://keras.io) neural network library. It exposes [command-line](http://openvax.github.io/mhcflurry/commandline_tutorial.html)
and [Python library](http://openvax.github.io/mhcflurry/python_tutorial.html) interfaces.

If you find MHCflurry useful in your research please cite:

> O'Donnell, T. et al., 2017. MHCflurry: open-source class I MHC binding affinity prediction. bioRxiv. Available at: http://www.biorxiv.org/content/early/2017/08/09/174243.

## Installation (pip)

Install the package:

```
$ pip install mhcflurry
```

Then download our datasets and trained models:

```
$ mhcflurry-downloads fetch
```

You can now generate predictions:

```
$ mhcflurry-predict \
       --alleles HLA-A0201 HLA-A0301 \
       --peptides SIINFEKL SIINFEKD SIINFEKQ \
       --out /tmp/predictions.csv
       
Wrote: /tmp/predictions.csv
```

See the [documentation](http://openvax.github.io/mhcflurry/) for more details.

## MHCflurry model variants and mass spec 

The default MHCflurry models are trained
on affinity measurements. Mass spec datasets are incorporated only in
the model selection step. We also release experimental predictors whose training data directly
includes mass spec. To download these predictors, run:

```
$ mhcflurry-downloads fetch models_class1_trained_with_mass_spec
```

and then to make them used by default:

```
$ export MHCFLURRY_DEFAULT_CLASS1_MODELS="$(mhcflurry-downloads path models_class1_trained_with_mass_spec)/models"
```

We also release predictors that do not use mass spec datasets at all. To use
these predictors, run:

```
$ mhcflurry-downloads fetch models_class1_selected_no_mass_spec
export MHCFLURRY_DEFAULT_CLASS1_MODELS="$(mhcflurry-downloads path models_class1_selected_no_mass_spec)/models"
```