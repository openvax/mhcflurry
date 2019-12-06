import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import pandas
import tempfile
import pickle

from numpy.testing import assert_, assert_equal, assert_allclose, assert_array_equal
from nose.tools import assert_greater, assert_less
import numpy

from mhcflurry import Class1AffinityPredictor
from mhcflurry.allele_encoding import MultipleAlleleEncoding
from mhcflurry.class1_presentation_neural_network import Class1PresentationNeuralNetwork
from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor
from mhcflurry.downloads import get_path
from mhcflurry.common import random_peptides
from mhcflurry.testing_utils import cleanup, startup
from mhcflurry.regression_target import to_ic50

AFFINITY_PREDICTOR = None

def setup():
    global AFFINITY_PREDICTOR
    startup()
    PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.no_mass_spec"),
        optimization_level=0,
        max_models=1)


def teardown():
    global AFFINITY_PREDICTOR
    PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = None
    cleanup()


def test_basic():
    affinity_predictor = AFFINITY_PREDICTOR
    models = []
    for affinity_network in affinity_predictor.class1_pan_allele_models:
        presentation_network = Class1PresentationNeuralNetwork(
            optimizer="adam",
            random_negative_rate=0.0,
            random_negative_constant=0,
            max_epochs=25,
            learning_rate=0.001,
            batch_generator_batch_size=256)
        presentation_network.load_from_class1_neural_network(affinity_network)
        models.append(presentation_network)

    predictor = Class1PresentationPredictor(
        models=models,
        allele_to_sequence=affinity_predictor.allele_to_sequence)

    alleles = ["HLA-A*02:01", "HLA-B*27:01", "HLA-C*07:02"]

    df = pandas.DataFrame(index=numpy.unique(random_peptides(1000, length=9)))
    for allele in alleles:
        df[allele] = affinity_predictor.predict(
            df.index.values, allele=allele)
    df["tightest_affinity"] = df.min(1)
    df["tightest_allele"] = df.idxmin(1)

    # Test untrained predictor gives predictions that match the affinity
    # predictor
    df_predictor = predictor.predict_to_dataframe(
        peptides=df.index.values,
        alleles=alleles)
    merged_df = pandas.merge(
        df, df_predictor.set_index("peptide"), left_index=True, right_index=True)

    print(merged_df)

    assert_allclose(
        merged_df["tightest_affinity"], merged_df["affinity"], rtol=1e-5)
    assert_allclose(
        merged_df["tightest_affinity"], to_ic50(merged_df["score"]), rtol=1e-5)
    assert_array_equal(merged_df["tightest_allele"], merged_df["allele"])

    # Test saving and loading
    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    predictor.save(models_dir)
    predictor2 = Class1PresentationPredictor.load(models_dir)

    df_predictor2 = predictor2.predict_to_dataframe(
        peptides=df.index.values,
        alleles=alleles)
    assert_array_equal(df_predictor.values, df_predictor2.values)

    # Test pickling
    predictor3 = pickle.loads(
        pickle.dumps(predictor, protocol=pickle.HIGHEST_PROTOCOL))
    predictor4 = pickle.loads(
        pickle.dumps(predictor2, protocol=pickle.HIGHEST_PROTOCOL))
    df_predictor3 = predictor3.predict_to_dataframe(
        peptides=df.index.values,
        alleles=alleles)
    df_predictor4 = predictor4.predict_to_dataframe(
        peptides=df.index.values,
        alleles=alleles)
    assert_array_equal(df_predictor.values, df_predictor3.values)
    assert_array_equal(df_predictor.values, df_predictor4.values)

    # Test that fitting a model changes the predictor but not the original model
    train_df = pandas.DataFrame({
        "peptide": numpy.concatenate([
            random_peptides(256, length=length)
            for length in [9]
        ]),
    })
    train_df["allele"] = "HLA-A*02:20"
    train_df["experiment"] = "experiment1"
    train_df["label"] = train_df.peptide.str.match("^[KSEQ]").astype("float32")
    train_df["pre_train_affinity_prediction"] = affinity_predictor.predict(
        train_df.peptide.values, alleles=train_df.allele.values)
    allele_encoding = MultipleAlleleEncoding(
        experiment_names=train_df.experiment.values,
        experiment_to_allele_list={
            "experiment1": ["HLA-A*02:20", "HLA-A*02:01"],
        },
        allele_to_sequence=predictor.allele_to_sequence)
    model = predictor4.models[0]
    new_predictor = Class1PresentationPredictor(
        models=[model],
        allele_to_sequence=predictor4.allele_to_sequence)
    train_df["original_score"] = new_predictor.predict(
        train_df.peptide.values, alleles=["HLA-A*02:20"])
    model.fit(
        peptides=train_df.peptide.values,
        labels=train_df.label.values,
        allele_encoding=allele_encoding)
    train_df["updated_score"] = new_predictor.predict(
        train_df.peptide.values,
        alleles=["HLA-A*02:20"])
    train_df["updated_affinity"] = new_predictor.predict_to_dataframe(
        train_df.peptide.values,
        alleles=["HLA-A*02:20"]).affinity.values
    train_df["score_diff"] = train_df.updated_score - train_df.original_score
    mean_change = train_df.groupby("label").score_diff.mean()
    print("Mean change:")
    print(mean_change)
    assert_greater(mean_change[1.0], mean_change[0.0])
    print(train_df)
    train_df["post_train_affinity_prediction"] = affinity_predictor.predict(
        train_df.peptide.values,
        alleles=train_df.allele.values)
    assert_array_equal(
        train_df.pre_train_affinity_prediction.values,
        train_df.post_train_affinity_prediction.values)

    (affinity_model,) = affinity_predictor.class1_pan_allele_models
    model.copy_weights_to_affinity_model(affinity_model)
    train_df["post_copy_weights_prediction"] = affinity_predictor.predict(
        train_df.peptide.values, alleles=train_df.allele.values)
    assert_allclose(
        train_df.updated_affinity.values,
        train_df.post_copy_weights_prediction.values,
        rtol=1e-5)
    train_df["affinity_diff"] = (
        train_df.post_copy_weights_prediction -
        train_df.post_train_affinity_prediction)
    median_affinity_change = train_df.groupby("label").affinity_diff.median()
    print("Median affinity change:")
    print(median_affinity_change)
    assert_less(median_affinity_change[1.0], median_affinity_change[0.0])

