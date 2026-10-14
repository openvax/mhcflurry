Python library tutorial
=======================

Predicting
----------

The MHCflurry Python API exposes additional options and features beyond those
supported by the commandline tools. This tutorial gives a basic overview
of the most important functionality. See the :ref:`API-documentation` for further details.

The `~mhcflurry.Class1AffinityPredictor` class is the primary user-facing interface.
Use the `~mhcflurry.Class1AffinityPredictor.load` static method to load a
trained predictor from disk. With no arguments this method will load the predictor
released with MHCflurry (see :ref:`downloading`\ ). If you pass a path to a
models directory, then it will load that predictor instead.

.. runblock:: pycon

    >>> from mhcflurry import Class1AffinityPredictor
    >>> predictor = Class1AffinityPredictor.load()
    >>> predictor.supported_alleles[:10]

With a predictor loaded we can now generate some binding predictions:

.. runblock:: pycon

    >>> predictor.predict(allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])

.. note::

    MHCflurry normalizes allele names using the `mhcnames <https://github.com/hammerlab/mhcnames>`__
    package. Names like ``HLA-A0201`` or ``A*02:01`` will be
    normalized to ``HLA-A*02:01``, so most naming conventions can be used
    with methods such as `~mhcflurry.Class1AffinityPredictor.predict`.

For more detailed results, we can use
`~mhcflurry.Class1AffinityPredictor.predict_to_dataframe`.

.. runblock:: pycon

    >>> predictor.predict_to_dataframe(allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])

Instead of a single allele and multiple peptides, we may need predictions for
allele/peptide pairs. We can predict across pairs by specifying
the `alleles` argument instead of `allele`. The list of alleles
must be the same length as the list of peptides (i.e. it is predicting over pairs,
*not* taking the cross product).

.. runblock:: pycon

    >>> predictor.predict(alleles=["HLA-A0201", "HLA-B*57:01"], peptides=["SIINFEKL", "SIINFEQL"])

Training
--------

Let's fit our own MHCflurry predictor. First we need some training data. If you
haven't already, run this in a shell to download the MHCflurry training data:

.. code-block:: shell

    $ mhcflurry-downloads fetch data_curated

We can get the path to this data from Python using `mhcflurry.downloads.get_path`:

.. runblock:: pycon

    >>> from mhcflurry.downloads import get_path
    >>> data_path = get_path("data_curated", "curated_training_data.no_mass_spec.csv.bz2")
    >>> data_path

Now let's load it with pandas and filter to reasonably-sized peptides:

.. runblock:: pycon

    >>> import pandas
    >>> df = pandas.read_csv(data_path)
    >>> df = df.loc[(df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 15)]
    >>> df.head(5)

We'll make an untrained `~mhcflurry.Class1AffinityPredictor` and then call
`~mhcflurry.Class1AffinityPredictor.fit_allele_specific_predictors` to fit
some models.

.. runblock:: pycon

    >>> new_predictor = Class1AffinityPredictor()
    >>> single_allele_train_data = df.loc[df.allele == "HLA-B*57:01"].sample(100)
    >>> new_predictor.fit_allele_specific_predictors(
    ...    n_models=1,
    ...    architecture_hyperparameters_list=[{
    ...         "layer_sizes": [16],
    ...         "max_epochs": 5,
    ...         "random_negative_constant": 5,
    ...    }],
    ...    peptides=single_allele_train_data.peptide.values,
    ...    affinities=single_allele_train_data.measurement_value.values,
    ...    allele="HLA-B*57:01")


The `~mhcflurry.Class1AffinityPredictor.fit_allele_specific_predictors` method
can be called any number of times on the same instance to build up ensembles
of models across alleles. The architecture hyperparameters we specified are
for demonstration purposes; to fit real models you would usually train for
more epochs.

Now we can generate predictions:

.. runblock:: pycon

    >>> new_predictor.predict(["SYNPEPII"], allele="HLA-B*57:01")

We can save our predictor to the specified directory on disk by running:

.. runblock:: pycon

    >>> new_predictor.save("/tmp/new-predictor")

and restore it:

.. runblock:: pycon

    >>> new_predictor2 = Class1AffinityPredictor.load("/tmp/new-predictor")
    >>> new_predictor2.supported_alleles


Lower level interface
---------------------

The high-level `Class1AffinityPredictor` delegates to low-level
`~mhcflurry.Class1NeuralNetwork` objects, each of which represents
a single neural network. The purpose of `~mhcflurry.Class1AffinityPredictor`
is to implement several important features:

ensembles
    More than one neural network can be used to generate each prediction. The
    predictions returned to the user are the geometric mean of the individual
    model predictions. This gives higher accuracy in most situations

multiple alleles
    A `~mhcflurry.Class1NeuralNetwork` generates predictions for only a single
    allele. The `~mhcflurry.Class1AffinityPredictor` maps alleles to the
    relevant `~mhcflurry.Class1NeuralNetwork` instances

serialization
    Loading and saving predictors is implemented in `~mhcflurry.Class1AffinityPredictor`.

Sometimes it's easiest to work directly with `~mhcflurry.Class1NeuralNetwork`.
Here is a simple example of doing so:

.. runblock:: pycon

    >>> from mhcflurry import Class1NeuralNetwork
    >>> network = Class1NeuralNetwork()
    >>> network.fit(
    ...    single_allele_train_data.peptide.values,
    ...    single_allele_train_data.measurement_value.values,
    ...    verbose=0)
    >>> network.predict(["SIINFEKLL"])

