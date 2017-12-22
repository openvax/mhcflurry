:orphan:

.. image:: https://travis-ci.org/hammerlab/mhcflurry.svg?branch=master
    :target: https://travis-ci.org/hammerlab/mhcflurry

.. image:: https://coveralls.io/repos/github/hammerlab/mhcflurry/badge.svg?branch=master
    :target: https://coveralls.io/github/hammerlab/mhcflurry

mhcflurry
===================

Open source neural network models for peptide-MHC binding affinity prediction

MHCflurry is an open source package for peptide/MHC I binding affinity
prediction. It provides competitive accuracy with a fast and
documented implementation.

You can download pre-trained MHCflurry models fit to affinity
measurements deposited in IEDB or train a MHCflurry predictor on your
own data.

Currently only allele-specific prediction is implemented, in which
separate models are trained for each allele. The released models
therefore support a fixed set of common class I alleles for which
sufficient published training data is available (see Supported alleles
and peptide lengths).

MHCflurry supports Python versions 2.7 and 3.4+. It uses the keras
neural network library via either the Tensorflow or Theano backends.
GPUs may optionally be used for a generally modest speed improvement.

If you find MHCflurry useful in your research please cite:

   O’Donnell, T. et al., 2017. MHCflurry: open-source class I MHC
   binding affinity prediction. bioRxiv. Available at:
   http://www.biorxiv.org/content/early/2017/08/09/174243.


Installation (pip)
******************

Install the package:

   $ pip install mhcflurry

Then download our datasets and trained models:

   $ mhcflurry-downloads fetch

From a checkout you can run the unit tests with:

   $ pip install nose
   $ nosetests .


Using conda
***********

You can alternatively get up and running with a conda environment as
follows. Some users have reported that this can avoid problems
installing tensorflow.

   $ conda create -q -n mhcflurry-env python=3.6 'tensorflow>=1.1.2'
   $ source activate mhcflurry-env

Then continue as above:

   $ pip install mhcflurry
   $ mhcflurry-downloads fetch


Command-line tutorial
=====================


Downloading models
******************

Most users will use pre-trained MHCflurry models that we release.
These models are distributed separately from the pip package and may
be downloaded with the mhcflurry-downloads tool:

   $ mhcflurry-downloads fetch models_class1

Files downloaded with mhcflurry-downloads are stored in a platform-
specific directory. To get the path to downloaded data, you can use:

   $ mhcflurry-downloads path models_class1
   /Users/tim/Library/Application Support/mhcflurry/4/1.0.0/models_class1/

We also release a few other “downloads,” such as curated training data
and some experimental models. To see what’s available and what you
have downloaded, run:

   $ mhcflurry-downloads info
   Environment variables
     MHCFLURRY_DATA_DIR                  [unset or empty]
     MHCFLURRY_DOWNLOADS_CURRENT_RELEASE [unset or empty]
     MHCFLURRY_DOWNLOADS_DIR             [unset or empty]

   Configuration
     current release                     = 1.0.0                
     downloads dir                       = '/Users/tim/Library/Application Support/mhcflurry/4/1.0.0' [exists]

   DOWNLOAD NAME                             DOWNLOADED?    DEFAULT?      URL                  
   models_class1                             YES            YES           http://github.com/hammerlab/mhcflurry/releases/download/pre-1.0/models_class1.tar.bz2 
   models_class1_experiments1                NO             NO            http://github.com/hammerlab/mhcflurry/releases/download/pre-1.0/models_class1_experiments1.tar.bz2 
   cross_validation_class1                   YES            NO            http://github.com/hammerlab/mhcflurry/releases/download/pre-1.0/cross_validation_class1.tar.bz2 
   data_iedb                                 NO             NO            https://github.com/hammerlab/mhcflurry/releases/download/pre-1.0/data_iedb.tar.bz2 
   data_kim2014                              NO             NO            http://github.com/hammerlab/mhcflurry/releases/download/0.9.1/data_kim2014.tar.bz2 
   data_curated                              YES            YES           https://github.com/hammerlab/mhcflurry/releases/download/pre-1.0/data_curated.tar.bz2

Note: The code we use for *generating* the downloads is in the
  "downloads_generation" directory in the repository.


Generating predictions
**********************

The mhcflurry-predict command generates predictions from the command-
line. By default it will use the pre-trained models you downloaded
above; other models can be used by specifying the "--models" argument.

Running:

   $ mhcflurry-predict
       --alleles HLA-A0201 HLA-A0301
       --peptides SIINFEKL SIINFEKD SIINFEKQ
       --out /tmp/predictions.csv
   Wrote: /tmp/predictions.csv

results in a file like this:

   $ cat /tmp/predictions.csv
   allele,peptide,mhcflurry_prediction,mhcflurry_prediction_low,mhcflurry_prediction_high,mhcflurry_prediction_percentile
   HLA-A0201,SIINFEKL,4899.04784343,2767.76365395,7269.68364294,6.5097875
   HLA-A0201,SIINFEKD,21050.420243,16834.6585914,24129.0460917,34.297175
   HLA-A0201,SIINFEKQ,21048.4726578,16736.5612549,24111.0131144,34.297175
   HLA-A0301,SIINFEKL,28227.2989092,24826.3079098,32714.285974,33.9512125
   HLA-A0301,SIINFEKD,30816.7212184,27685.5084708,36037.3259046,41.225775
   HLA-A0301,SIINFEKQ,24183.0210465,19346.154182,32263.7124753,24.8109625

The predictions are given as affinities (KD) in nM in the
"mhcflurry_prediction" column. The other fields give the 5-95
percentile predictions across the models in the ensemble and the
quantile of the affinity prediction among a large number of random
peptides tested on that allele.

The predictions shown above were generated with MHCflurry 1.0.0.
Different versions of MHCflurry can give considerably different
results. Even on the same version, exact predictions may vary (up to
about 1 nM) depending on the Keras backend and other details.

In most cases you’ll want to specify the input as a CSV file instead
of passing peptides and alleles as commandline arguments. See
mhcflurry-predict docs.


Fitting your own models
***********************

The mhcflurry-class1-train-allele-specific-models command is used to
fit models to training data. The models we release with MHCflurry are
trained with a command like:

   $ mhcflurry-class1-train-allele-specific-models \
       --data TRAINING_DATA.csv \
       --hyperparameters hyperparameters.yaml \
       --percent-rank-calibration-num-peptides-per-length 1000000 \
       --min-measurements-per-allele 75 \
       --out-models-dir models

MHCflurry predictors are serialized to disk as many files in a
directory. The command above will write the models to the output
directory specified by the "--out-models-dir" argument. This directory
has files like:

   manifest.csv
   percent_ranks.csv
   weights_BOLA-6*13:01-0-1e6e7c0610ac68f8.npz
   ...
   weights_PATR-B*24:01-0-e12e0ee723833110.npz
   weights_PATR-B*24:01-0-ec4a36529321d868.npz
   weights_PATR-B*24:01-0-fd5a340098d3a9f4.npz

The "manifest.csv" file gives metadata for all the models used in the
predictor. There will be a "weights_..." file for each model giving
its weights (the parameters for the neural network). The
"percent_ranks.csv" stores a histogram of model predictions for each
allele over a large number of random peptides. It is used for
generating the percent ranks at prediction time.

To call mhcflurry-class1-train-allele-specific-models you’ll need some
training data. The data we use for our released predictors can be
downloaded with mhcflurry-downloads:

   $ mhcflurry-downloads fetch data_curated

It looks like this:

   $ bzcat "$(mhcflurry-downloads path data_curated)/curated_training_data.csv.bz2" | head -n 3
   allele,peptide,measurement_value,measurement_type,measurement_source,original_allele
   BoLA-1*21:01,AENDTLVVSV,7817.0,quantitative,Barlow - purified MHC/competitive/fluorescence,BoLA-1*02101
   BoLA-1*21:01,NQFNGGCLLV,1086.0,quantitative,Barlow - purified MHC/direct/fluorescence,BoLA-1*02101


Scanning protein sequences for predicted epitopes
*************************************************

The mhctools package provides support for scanning protein sequences
to find predicted epitopes. It supports MHCflurry as well as other
binding predictors. Here is an example.

First, install "mhctools" if it is not already installed:

   $ pip install mhctools

We’ll generate predictions across "example.fasta", a FASTA file with
two short sequences:

   >protein1
   MDSKGSSQKGSRLLLLLVVSNLLLCQGVVSTPVCPNGPGNCQV
   EMFNEFDKRYAQGKGFITMALNSCHTSSLPTPEDKEQAQQTHH
   >protein2
   VTEVRGMKGAPDAILSRAIEIEEENKRLLEGMEMIFGQVIPGA
   ARYSAFYNLLHCLRRDSSKIDTYLKLLNCRIIYNNNC

Here’s the "mhctools" invocation. See "mhctools -h" for more
information.

   $ mhctools
       --mhc-predictor mhcflurry
       --input-fasta-file example.fasta
       --mhc-alleles A02:01,A03:01
       --mhc-peptide-lengths 8,9,10,11
       --extract-subsequences
       --output-csv /tmp/subsequence_predictions.csv
   2017-12-22 01:12:44,974 - mhctools.cli.args - INFO - Building MHC binding prediction type for alleles ['HLA-A*02:01', 'HLA-A*03:01'] and epitope lengths [8, 9, 10, 11]
   2017-12-22 01:12:48,868 - mhctools.mhcflurry - INFO - BindingPrediction(peptide='AARYSAFY', allele='HLA-A*03:01', affinity=5744.3443, percentile_rank=None, source_sequence_name=None, offset=0, prediction_method_name='mhcflurry')
   ...

   [1192 rows x 8 columns]

This will write a file giving predictions for all subsequences of the
specified lengths:

   $ head -n 3 /tmp/subsequence_predictions.csv
   ,source_sequence_name,offset,peptide,allele,affinity,percentile_rank,prediction_method_name,length
   0,protein2,42,AARYSAFY,HLA-A*03:01,5744.3442744,,mhcflurry,8
   1,protein2,42,AARYSAFYN,HLA-A*03:01,10576.5364408,,mhcflurry,9


Python library tutorial
=======================


Predicting
**********

The MHCflurry Python API exposes additional options and features
beyond those supported by the commandline tools. This tutorial gives a
basic overview of the most important functionality. See the API
Documentation for further details.

The "Class1AffinityPredictor" class is the primary user-facing
interface. Use the "load" static method to load a trained predictor
from disk. With no arguments this method will load the predictor
released with MHCflurry (see Downloading models). If you pass a path
to a models directory, then it will load that predictor instead.

   >>> from mhcflurry import Class1AffinityPredictor
   >>> predictor = Class1AffinityPredictor.load()
   >>> predictor.supported_alleles[:10]
   ['BoLA-6*13:01', 'Eqca-1*01:01', 'H-2-Db', 'H-2-Dd', 'H-2-Kb', 'H-2-Kd', 'H-2-Kk', 'H-2-Ld', 'HLA-A*01:01', 'HLA-A*02:01']

With a predictor loaded we can now generate some binding predictions:

   >>> predictor.predict(allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])
   /Users/tim/miniconda3/envs/py2k/lib/python2.7/site-packages/h5py/__init__.py:34: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 80
     from ._conv import register_converters as _register_converters
   /Users/tim/miniconda3/envs/py2k/lib/python2.7/site-packages/h5py/__init__.py:43: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 80
     from . import h5a, h5d, h5ds, h5f, h5fd, h5g, h5r, h5s, h5t, h5p, h5z
   /Users/tim/miniconda3/envs/py2k/lib/python2.7/site-packages/h5py/_hl/group.py:21: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 80
     from .. import h5g, h5i, h5o, h5r, h5t, h5l, h5p
   Using TensorFlow backend.
   array([ 4899.04784343,  5685.25682682])

Note: MHCflurry normalizes allele names using the mhcnames package.
  Names like "HLA-A0201" or "A*02:01" will be normalized to
  "HLA-A*02:01", so most naming conventions can be used with methods
  such as "predict".

For more detailed results, we can use "predict_to_dataframe".

   >>> predictor.predict_to_dataframe(allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])
         allele   peptide   prediction  prediction_low  prediction_high  \
   0  HLA-A0201  SIINFEKL  4899.047843     2767.763654      7269.683643   
   1  HLA-A0201  SIINFEQL  5685.256827     3815.923563      7476.714466   

      prediction_percentile  
   0               6.509787  
   1               7.436687  

Instead of a single allele and multiple peptides, we may need
predictions for allele/peptide pairs. We can predict across pairs by
specifying the "alleles" argument instead of "allele". The list of
alleles must be the same length as the list of peptides (i.e. it is
predicting over pairs, *not* taking the cross product).

   >>> predictor.predict(alleles=["HLA-A0201", "HLA-B*57:01"], peptides=["SIINFEKL", "SIINFEQL"])
   array([  4899.04794216,  26704.22011499])


Training
********

Let’s fit our own MHCflurry predictor. First we need some training
data. If you haven’t already, run this in a shell to download the
MHCflurry training data:

   $ mhcflurry-downloads fetch data_curated

We can get the path to this data from Python using
"mhcflurry.downloads.get_path":

   >>> from mhcflurry.downloads import get_path
   >>> data_path = get_path("data_curated", "curated_training_data.csv.bz2")
   >>> data_path
   '/Users/tim/Library/Application Support/mhcflurry/4/1.0.0/data_curated/curated_training_data.csv.bz2'

Now let’s load it with pandas and filter to reasonably-sized peptides:

   >>> import pandas
   >>> df = pandas.read_csv(data_path)
   >>> df = df.loc[(df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)]
   >>> df.head(5)
            allele     peptide  measurement_value measurement_type  \
   0  BoLA-1*21:01  AENDTLVVSV             7817.0     quantitative   
   1  BoLA-1*21:01  NQFNGGCLLV             1086.0     quantitative   
   2  BoLA-2*08:01   AAHCIHAEW               21.0     quantitative   
   3  BoLA-2*08:01   AAKHMSNTY             1299.0     quantitative   
   4  BoLA-2*08:01  DSYAYMRNGW                2.0     quantitative   

                                  measurement_source original_allele  
   0  Barlow - purified MHC/competitive/fluorescence    BoLA-1*02101  
   1       Barlow - purified MHC/direct/fluorescence    BoLA-1*02101  
   2       Barlow - purified MHC/direct/fluorescence    BoLA-2*00801  
   3       Barlow - purified MHC/direct/fluorescence    BoLA-2*00801  
   4       Barlow - purified MHC/direct/fluorescence    BoLA-2*00801  

We’ll make an untrained "Class1AffinityPredictor" and then call
"fit_allele_specific_predictors" to fit some models.

   >>> new_predictor = Class1AffinityPredictor()
   >>> single_allele_train_data = df.loc[df.allele == "HLA-B*57:01"].sample(100)
   >>> new_predictor.fit_allele_specific_predictors(
   ...    n_models=1,
   ...    architecture_hyperparameters={
   ...         "layer_sizes": [16],
   ...         "max_epochs": 5,
   ...         "random_negative_constant": 5,
   ...    },
   ...    peptides=single_allele_train_data.peptide.values,
   ...    affinities=single_allele_train_data.measurement_value.values,
   ...    allele="HLA-B*57:01")
   Train on 112 samples, validate on 28 samples
   Epoch 1/1

   112/112 [==============================] - 0s 3ms/step - loss: 0.3730 - val_loss: 0.3472
   Epoch   0 /   5: loss=0.373015. Min val loss (None) at epoch None
   Train on 112 samples, validate on 28 samples
   Epoch 1/1

   112/112 [==============================] - 0s 38us/step - loss: 0.3508 - val_loss: 0.3345
   Train on 112 samples, validate on 28 samples
   Epoch 1/1

   112/112 [==============================] - 0s 37us/step - loss: 0.3375 - val_loss: 0.3218
   Train on 112 samples, validate on 28 samples
   Epoch 1/1

   112/112 [==============================] - 0s 36us/step - loss: 0.3227 - val_loss: 0.3092
   Train on 112 samples, validate on 28 samples
   Epoch 1/1

   112/112 [==============================] - 0s 37us/step - loss: 0.3104 - val_loss: 0.2970
   [<mhcflurry.class1_neural_network.Class1NeuralNetwork object at 0x11e28ad10>]

The "fit_allele_specific_predictors" method can be called any number
of times on the same instance to build up ensembles of models across
alleles. The "architecture_hyperparameters" we specified are for
demonstration purposes; to fit real models you would usually train for
more epochs.

Now we can generate predictions:

   >>> new_predictor.predict(["SYNPEPII"], allele="HLA-B*57:01")
   array([ 610.30706541])

We can save our predictor to the specified directory on disk by
running:

   >>> new_predictor.save("/tmp/new-predictor")

and restore it:

   >>> new_predictor2 = Class1AffinityPredictor.load("/tmp/new-predictor")
   >>> new_predictor2.supported_alleles
   ['HLA-B*57:01']


Lower level interface
*********************

The high-level "Class1AffinityPredictor" delegates to low-level
"Class1NeuralNetwork" objects, each of which represents a single
neural network. The purpose of "Class1AffinityPredictor" is to
implement several important features:

ensembles
   More than one neural network can be used to generate each
   prediction. The predictions returned to the user are the geometric
   mean of the individual model predictions. This gives higher
   accuracy in most situations

multiple alleles
   A "Class1NeuralNetwork" generates predictions for only a single
   allele. The "Class1AffinityPredictor" maps alleles to the relevant
   "Class1NeuralNetwork" instances

serialization
   Loading and saving predictors is implemented in
   "Class1AffinityPredictor".

Sometimes it’s easiest to work directly with "Class1NeuralNetwork".
Here is a simple example of doing so:

   >>> from mhcflurry import Class1NeuralNetwork
   >>> network = Class1NeuralNetwork()
   >>> network.fit(
   ...    single_allele_train_data.peptide.values,
   ...    single_allele_train_data.measurement_value.values,
   ...    verbose=0)
   Epoch   0 / 500: loss=0.533378. Min val loss (None) at epoch None
   Early stopping at epoch 124 / 500: loss=0.0115427. Min val loss (0.0719302743673) at epoch 113
   >>> network.predict(["SIINFEKLL"])
   array([ 23004.58985458])


Supported alleles and peptide lengths
=====================================

Models released with the current version of MHCflurry (1.0.0) support
peptides of length 8-15 and the following 124 alleles:

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


mhcflurry
=========

Open source neural network models for peptide-MHC binding affinity
prediction

The adaptive immune system depends on the presentation of protein
fragments by MHC molecules. Machine learning models of this
interaction are used in studies of infectious diseases, autoimmune
diseases, vaccine development, and cancer immunotherapy.

MHCflurry supports Class I peptide/MHC binding affinity prediction
using ensembles of allele-specific models. You can fit MHCflurry
models to your own data or download models that we fit to data from
IEDB and Kim 2014. Our combined dataset is available for download
here.

Pan-allelic prediction is supported in principle but is not yet
performing accurately. Infrastructure for modeling other aspects of
antigen processing is also implemented but experimental.

If you find MHCflurry useful in your research please cite:

   O’Donnell, T. et al., 2017. MHCflurry: open-source class I MHC
   binding affinity prediction. bioRxiv. Available at:
   http://www.biorxiv.org/content/early/2017/08/09/174243.


Setup (pip)
***********

Install the package:

   pip install mhcflurry

Then download our datasets and trained models:

   mhcflurry-downloads fetch

From a checkout you can run the unit tests with:

   nosetests .

The MHCflurry predictors are implemented in Python using keras.

MHCflurry works with both the tensorflow and theano keras backends.
The tensorflow backend gives faster model-loading time but is
undergoing more rapid development and sometimes hits issues. If you
encounter tensorflow errors running MHCflurry, try setting this
environment variable to switch to the theano backend:

   export KERAS_BACKEND=theano

You may also needs to "pip install theano".


Setup (conda)
*************

You can alternatively get up and running with a conda environment as
follows:

   conda create -q -n mhcflurry-env python=3.6 'tensorflow>=1.1.0'
   source activate mhcflurry-env

Then continue as above:

   pip install mhcflurry
   mhcflurry-downloads fetch

If you wish to test your installation, you can install "nose" and run
the tests from a checkout:

   pip install nose
   nosetests .


Making predictions from the command-line
****************************************

   $ mhcflurry-predict --alleles HLA-A0201 HLA-A0301 --peptides SIINFEKL SIINFEKD SIINFEKQ
   allele,peptide,mhcflurry_prediction,mhcflurry_prediction_low,mhcflurry_prediction_high
   HLA-A0201,SIINFEKL,5326.541919062165,3757.86675352994,7461.37693353508
   HLA-A0201,SIINFEKD,18763.70298522213,13140.82000240037,23269.82139560844
   HLA-A0201,SIINFEKQ,18620.10057358322,13096.425874678192,23223.148184869413
   HLA-A0301,SIINFEKL,24481.726678691946,21035.52779725433,27245.371837497867
   HLA-A0301,SIINFEKD,24687.529360239587,21582.590014592537,27749.39869616437
   HLA-A0301,SIINFEKQ,25923.062203902562,23522.5793450799,28079.456657427705

The predictions returned are affinities (KD) in nM. The
"prediction_low" and "prediction_high" fields give the 5-95 percentile
predictions across the models in the ensemble. The predictions above
were generated with MHCflurry 0.9.2. Your exact predictions may vary
slightly from these (up to about 1 nM) depending on the Keras backend
in use and other numerical details. Different versions of MHCflurry
can of course give results considerably different from these.

You can also specify the input and output as CSV files. Run
"mhcflurry-predict -h" for details.


Making predictions from Python
******************************

   >>> from mhcflurry import Class1AffinityPredictor
   >>> predictor = Class1AffinityPredictor.load()
   >>> predictor.predict_to_dataframe(peptides=['SIINFEKL'], allele='A0201')


     allele   peptide   prediction  prediction_low  prediction_high
     A0201  SIINFEKL  6029.084473     4474.103253      7771.297702

See the class1_allele_specific_models.ipynb notebook for an overview
of the Python API, including fitting your own predictors.


Scanning protein sequences for predicted epitopes
*************************************************

The mhctools package provides support for scanning protein sequences
to find predicted epitopes. It supports MHCflurry as well as other
binding predictors. Here is an example:

   # First install mhctools if needed:
   pip install mhctools

   # Now generate predictions for protein sequences in FASTA format:
   mhctools \
       --mhc-predictor mhcflurry \
       --input-fasta-file INPUT.fasta \
       --mhc-alleles A02:01,A03:01 \
       --mhc-peptide-lengths 8,9,10,11 \
       --extract-subsequences \
       --out RESULT.csv


Details on the downloadable models
**********************************


Environment variables
*********************

The path where MHCflurry looks for model weights and data can be set
with the "MHCFLURRY_DOWNLOADS_DIR" environment variable. This
directory should contain subdirectories like “models_class1”.
