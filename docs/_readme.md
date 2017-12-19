[![Build Status](https://travis-ci.org/hammerlab/mhcflurry.svg?branch=master)](https://travis-ci.org/hammerlab/mhcflurry) [![Coverage Status](https://coveralls.io/repos/github/hammerlab/mhcflurry/badge.svg?branch=master)](https://coveralls.io/github/hammerlab/mhcflurry?branch=master)

# mhcflurry
Open source peptide/MHC I binding affinity prediction

<!-- README.MD IS AUTO-GENERATED -->
<!-- DO NOT EDIT README.md, EDIT FILES in docs/ INSTEAD -->
<!-- Then run "make rst" in the docs/ directory to regenerate -->

Introduction and setup
----------------------

MHCflurry is a peptide/MHC I binding affinity prediction package written in Python. It aims to provide state of the art accuracy with a documented, fast, and open source implementation.

MHCflurry users may download trained predictors fit to affinity measurements deposited in IEDB. See the "downloads\_generation/models\_class1" directory in the repository for the workflow used to train these predictors. It is also easy for users with their own data to fit their own models.

Currently only allele-specific prediction is implemented, in which separate models are trained for each allele. The released models therefore support a fixed set of common class I alleles for which sufficient published training data is available.

MHCflurry supports Python versions 2.7 and 3.4+. It uses the Keras neural network library via either the Tensorflow or Theano backends. GPUs may optionally be used for a generally modest speed improvement.

If you find MHCflurry useful in your research please cite:

> O'Donnell, T. et al., 2017. MHCflurry: open-source class I MHC binding affinity prediction. bioRxiv. Available at: <http://www.biorxiv.org/content/early/2017/08/09/174243>.

### Installation (pip)

Install the package:

    pip install mhcflurry

Then download our datasets and trained models:

    mhcflurry-downloads fetch

From a checkout you can run the unit tests with:

    pip install nose
    nosetests .

### Using conda

You can alternatively get up and running with a [conda](https://conda.io/docs/) environment as follows. Some users have reported that this can avoid problems installing tensorflow.

    conda create -q -n mhcflurry-env python=3.6 'tensorflow>=1.1.2'
    source activate mhcflurry-env

Then continue as above:

    pip install mhcflurry
    mhcflurry-downloads fetch

Using MHCflurry from the command-line
-------------------------------------

### mhcflurry-predict

The `mhcflurry-predict` command generates predictions from the command-line.

``` sourceCode
$ mhcflurry-predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD SIINFEKQ
allele,peptide,mhcflurry_prediction,mhcflurry_prediction_low,mhcflurry_prediction_high
HLA-A0201,SIINFEKL,5326.541919062165,3757.86675352994,7461.37693353508
HLA-A0201,SIINFEKD,18763.70298522213,13140.82000240037,23269.82139560844
HLA-A0201,SIINFEKQ,18620.10057358322,13096.425874678192,23223.148184869413
HLA-A0301,SIINFEKL,24481.726678691946,21035.52779725433,27245.371837497867
HLA-A0301,SIINFEKD,24687.529360239587,21582.590014592537,27749.39869616437
HLA-A0301,SIINFEKQ,25923.062203902562,23522.5793450799,28079.456657427705
```

The predictions returned are affinities (KD) in nM. The `prediction_low` and `prediction_high` fields give the 5-95 percentile predictions across the models in the ensemble. The predictions above were generated with MHCflurry 0.9.2.

Your exact predictions may vary slightly from these (up to about 1 nM) depending on the Keras backend in use and other numerical details. Different versions of MHCflurry can of course give results considerably different from these.

You can also specify the input and output as CSV files. Run `mhcflurry-predict -h` for details.

Using MHCflurry as a library
----------------------------

xxx

Supported peptides and alleles
------------------------------

Models released with the current version of MHCflurry (1.0.0) support peptides of length 8-15 and the following 124 alleles:

    BoLA-6*13:01, Eqca-1*01:01, H-2-Db, H-2-Dd, H-2-Kb, H-2-Kd, H-2-Kk,
    H-2-Ld, HLA-A*01:01, HLA-A*02:01, HLA-A*02:02, HLA-A*02:03,
    HLA-A*02:05, HLA-A*02:06, HLA-A*02:07, HLA-A*02:11, HLA-A*02:12,
    HLA-A*02:16, HLA-A*02:17, HLA-A*02:19, HLA-A*02:50, HLA-A*03:01,
    HLA-A*11:01, HLA-A*23:01, HLA-A*24:01, HLA-A*24:02, HLA-A*24:03,
    HLA-A*25:01, HLA-A*26:01, HLA-A*26:02, HLA-A*26:03, HLA-A*29:02,
    HLA-A*30:01, HLA-A*30:02, HLA-A*31:01, HLA-A*32:01, HLA-A*32:07,
    HLA-A*33:01, HLA-A*66:01, HLA-A*68:01, HLA-A*68:02, HLA-A*68:23,
    HLA-A*69:01, HLA-A*80:01, HLA-B*07:01, HLA-B*07:02, HLA-B*08:01,
    HLA-B*08:02, HLA-B*08:03, HLA-B*14:02, HLA-B*15:01, HLA-B*15:02,
    HLA-B*15:03, HLA-B*15:09, HLA-B*15:17, HLA-B*15:42, HLA-B*18:01,
    HLA-B*27:01, HLA-B*27:03, HLA-B*27:04, HLA-B*27:05, HLA-B*27:06,
    HLA-B*27:20, HLA-B*35:01, HLA-B*35:03, HLA-B*35:08, HLA-B*37:01,
    HLA-B*38:01, HLA-B*39:01, HLA-B*40:01, HLA-B*40:02, HLA-B*42:01,
    HLA-B*44:01, HLA-B*44:02, HLA-B*44:03, HLA-B*45:01, HLA-B*45:06,
    HLA-B*46:01, HLA-B*48:01, HLA-B*51:01, HLA-B*53:01, HLA-B*54:01,
    HLA-B*57:01, HLA-B*58:01, HLA-B*73:01, HLA-B*83:01, HLA-C*03:03,
    HLA-C*03:04, HLA-C*04:01, HLA-C*05:01, HLA-C*06:02, HLA-C*07:01,
    HLA-C*07:02, HLA-C*08:02, HLA-C*12:03, HLA-C*14:02, HLA-C*15:02,
    Mamu-A*01:01, Mamu-A*02:01, Mamu-A*02:0102, Mamu-A*07:01,
    Mamu-A*07:0103, Mamu-A*11:01, Mamu-A*22:01, Mamu-A*26:01,
    Mamu-B*01:01, Mamu-B*03:01, Mamu-B*08:01, Mamu-B*10:01, Mamu-B*17:01,
    Mamu-B*17:04, Mamu-B*39:01, Mamu-B*52:01, Mamu-B*66:01, Mamu-B*83:01,
    Mamu-B*87:01, Patr-A*01:01, Patr-A*03:01, Patr-A*04:01, Patr-A*07:01,
    Patr-A*09:01, Patr-B*01:01, Patr-B*13:01, Patr-B*24:01
