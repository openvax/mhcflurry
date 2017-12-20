Library usage
=============

.. runblock:: pycon

    >>> # Load downloaded predictor
    >>> import mhcflurry
    >>> predictor = mhcflurry.Class1AffinityPredictor.load()
    >>> print(predictor.supported_alleles)


::


    # coding: utf-8

    # In[22]:

    import pandas
    import numpy
    import seaborn
    import logging
    from matplotlib import pyplot

    import mhcflurry

    print("MHCflurry version: %s" % (mhcflurry.__version__))


    # # Download data and models

    # In[2]:

    get_ipython().system('mhcflurry-downloads fetch')


    # # Making predictions with `Class1AffinityPredictor`

    # In[3]:

    help(mhcflurry.Class1AffinityPredictor)


    # In[4]:

    downloaded_predictor = mhcflurry.Class1AffinityPredictor.load()


    # In[5]:

    downloaded_predictor.predict(allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])


    # In[6]:

    downloaded_predictor.predict_to_dataframe(allele="HLA-A0201", peptides=["SIINFEKL", "SIINFEQL"])


    # In[7]:

    downloaded_predictor.predict_to_dataframe(alleles=["HLA-A0201", "HLA-B*57:01"], peptides=["SIINFEKL", "SIINFEQL"])


    # In[8]:

    downloaded_predictor.predict_to_dataframe(
        allele="HLA-A0201",
        peptides=["SIINFEKL", "SIINFEQL"],
        include_individual_model_predictions=True)


    # In[9]:

    downloaded_predictor.predict_to_dataframe(
        allele="HLA-A0201",
        peptides=["SIINFEKL", "SIINFEQL", "TAAAALANGGGGGGGG"],
        throw=False)  # Without throw=False, you'll get a ValueError for invalid peptides or alleles


    # # Instantiating a `Class1AffinityPredictor`  from a saved model on disk

    # In[10]:

    models_dir = mhcflurry.downloads.get_path("models_class1", "models")
    models_dir


    # In[11]:

    # This will be the same predictor we instantiated above. We're just being explicit about what models to load.
    downloaded_predictor = mhcflurry.Class1AffinityPredictor.load(models_dir)
    downloaded_predictor.predict(["SIINFEKL", "SIQNPEKP", "SYNFPEPI"], allele="HLA-A0301")


    # # Fit a model: first load some data

    # In[12]:

    # This is the data the downloaded models were trained on
    data_path = mhcflurry.downloads.get_path("data_curated", "curated_training_data.csv.bz2")
    data_path


    # In[13]:

    data_df = pandas.read_csv(data_path)
    data_df


    # # Fit a model: Low level `Class1NeuralNetwork` interface

    # In[14]:

    # We'll use mostly the default hyperparameters here. Could also specify them as kwargs.
    new_model = mhcflurry.Class1NeuralNetwork(layer_sizes=[16])
    new_model.hyperparameters


    # In[16]:

    train_data = data_df.loc[
        (data_df.allele == "HLA-B*57:01") &
        (data_df.peptide.str.len() >= 8) &
        (data_df.peptide.str.len() <= 15)
    ]
    get_ipython().magic('time new_model.fit(train_data.peptide.values, train_data.measurement_value.values)')


    # In[17]:

    new_model.predict(["SYNPEPII"])


    # # Fit a model: high level `Class1AffinityPredictor` interface

    # In[18]:

    affinity_predictor = mhcflurry.Class1AffinityPredictor()

    # This can be called any number of times, for example on different alleles, to build up the ensembles.
    affinity_predictor.fit_allele_specific_predictors(
        n_models=1,
        architecture_hyperparameters={"layer_sizes": [16], "max_epochs": 10},
        peptides=train_data.peptide.values,
        affinities=train_data.measurement_value.values,
        allele="HLA-B*57:01",
    )


    # In[19]:

    affinity_predictor.predict(["SYNPEPII"], allele="HLA-B*57:01")


    # # Save and restore the fit model

    # In[20]:

    get_ipython().system('mkdir /tmp/saved-affinity-predictor')
    affinity_predictor.save("/tmp/saved-affinity-predictor")
    get_ipython().system('ls /tmp/saved-affinity-predictor')


    # In[21]:

    affinity_predictor2 = mhcflurry.Class1AffinityPredictor.load("/tmp/saved-affinity-predictor")
    affinity_predictor2.predict(["SYNPEPII"], allele="HLA-B*57:01")

