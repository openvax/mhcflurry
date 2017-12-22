[![Build Status](https://travis-ci.org/hammerlab/mhcflurry.svg?branch=master)](https://travis-ci.org/hammerlab/mhcflurry) [![Coverage Status](https://coveralls.io/repos/github/hammerlab/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/mhcflurry?branch=master)

# mhcflurry
[MHC I](https://en.wikipedia.org/wiki/Major_histocompatibility_complex) ligand
prediction package with competitive accuracy and a fast, easily installed, and 
[documented](http://www.hammerlab.org/mhcflurry/) open source codebase.

MHCflurry supports Class I peptide/MHC binding affinity prediction using
ensembles of allele-specific models. You can fit MHCflurry models to your own data
or download models that we fit to data from
[IEDB](http://www.iedb.org/home_v3.php) and [Kim 2014](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-241).

MHCflurry runs on Python versions 2.7 and 3.4+. It uses the [keras](https://keras.io)
neural network library via either the Tensorflow or Theano backends. GPUs may
optionally be used for a generally modest speed improvement.

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
       --out /tmp/predictions.csv \
       
Wrote: /tmp/predictions.csv
```

See the [documentation](http://www.hammerlab.org/mhcflurry/) for more details.
