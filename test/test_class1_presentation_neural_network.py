import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import pandas
import argparse
import sys
import copy
import os
import tempfile
import pickle

from numpy.testing import assert_, assert_equal, assert_allclose, assert_array_equal
from nose.tools import assert_greater, assert_less
import numpy
from random import shuffle

from sklearn.metrics import roc_auc_score

from mhcflurry import Class1AffinityPredictor
from mhcflurry.allele_encoding import MultipleAlleleEncoding
from mhcflurry.class1_presentation_neural_network import Class1PresentationNeuralNetwork
from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.downloads import get_path
from mhcflurry.regression_target import from_ic50
from mhcflurry.common import random_peptides, positional_frequency_matrix
from mhcflurry.testing_utils import cleanup, startup
from mhcflurry.amino_acid import COMMON_AMINO_ACIDS
from mhcflurry.custom_loss import MultiallelicMassSpecLoss
from mhcflurry.regression_target import to_ic50


###################################################
# SETUP
###################################################

COMMON_AMINO_ACIDS = sorted(COMMON_AMINO_ACIDS)

AFFINITY_PREDICTOR = None

def setup():
    global AFFINITY_PREDICTOR
    startup()
    AFFINITY_PREDICTOR = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.no_mass_spec"),
        optimization_level=0,
        max_models=1)


def teardown():
    global AFFINITY_PREDICTOR
    AFFINITY_PREDICTOR = None
    cleanup()


def data_path(name):
    '''
    Return the absolute path to a file in the test/data directory.
    The name specified should be relative to test/data.
    '''
    return os.path.join(os.path.dirname(__file__), "data", name)


###################################################
# UTILITY FUNCTIONS
###################################################

def scramble_peptide(peptide):
    lst = list(peptide)
    shuffle(lst)
    return "".join(lst)


def make_motif(presentation_predictor, allele, peptides, frac=0.01, master_allele_encoding=None):
    if master_allele_encoding is not None:
        alleles = MultipleAlleleEncoding(borrow_from=master_allele_encoding)
        alleles.append_alleles([allele] * len(peptides))
    else:
        alleles = [allele]

    peptides = EncodableSequences.create(peptides)
    predictions = presentation_predictor.predict(
        peptides=peptides,
        alleles=alleles,
    )
    random_predictions_df = pandas.DataFrame({"peptide": peptides.sequences})
    random_predictions_df["prediction"] = predictions
    random_predictions_df = random_predictions_df.sort_values(
        "prediction", ascending=False)
    top = random_predictions_df.iloc[:int(len(random_predictions_df) * frac)]
    matrix = positional_frequency_matrix(top.peptide.values)
    return matrix


###################################################
# TESTS
###################################################

def Xtest_synthetic_allele_refinement_max_affinity(include_affinities=True):
    """
    Test that in a synthetic example the model is able to learn that HLA-C*01:02
    prefers P at position 3.
    """
    refine_allele = "HLA-C*01:02"
    alleles = ["HLA-A*02:01", "HLA-B*27:01", "HLA-C*07:01", "HLA-A*03:01",
        "HLA-B*15:01", refine_allele]
    peptides_per_allele = [2000, 1000, 500, 1500, 1200, 800, ]

    allele_to_peptides = dict(zip(alleles, peptides_per_allele))

    length = 9

    train_with_ms = pandas.read_csv(get_path("data_curated",
        "curated_training_data.with_mass_spec.csv.bz2"))
    train_no_ms = pandas.read_csv(
        get_path("data_curated", "curated_training_data.no_mass_spec.csv.bz2"))

    def filter_df(df):
        return df.loc[
            (df.allele.isin(alleles)) & (df.peptide.str.len() == length)]

    train_with_ms = filter_df(train_with_ms)
    train_no_ms = filter_df(train_no_ms)

    ms_specific = train_with_ms.loc[
        ~train_with_ms.peptide.isin(train_no_ms.peptide)]

    train_peptides = []
    train_true_alleles = []
    for allele in alleles:
        peptides = ms_specific.loc[ms_specific.allele == allele].peptide.sample(
            n=allele_to_peptides[allele])
        train_peptides.extend(peptides)
        train_true_alleles.extend([allele] * len(peptides))

    hits_df = pandas.DataFrame({"peptide": train_peptides})
    hits_df["true_allele"] = train_true_alleles
    hits_df["hit"] = 1.0
    hits_df["label"] = 500
    hits_df["measurement_inequality"] = "<"

    decoys_df = hits_df.copy()
    decoys_df["peptide"] = decoys_df.peptide.map(scramble_peptide)
    decoys_df["true_allele"] = ""
    decoys_df["hit"] = 0.0
    decoys_df["label"] = 500
    hits_df["measurement_inequality"] = ">"

    mms_train_df = pandas.concat([hits_df, decoys_df], ignore_index=True)
    mms_train_df["label"] = mms_train_df.hit
    mms_train_df["is_affinity"] = True

    if include_affinities:
        affinity_train_df = pandas.read_csv(get_path("models_class1_pan",
            "models.with_mass_spec/train_data.csv.bz2"))
        affinity_train_df = affinity_train_df.loc[
            affinity_train_df.allele.isin(alleles),
            ["peptide", "allele", "measurement_inequality", "measurement_value"]
        ]

        affinity_train_df["label"] = affinity_train_df["measurement_value"]
        del affinity_train_df["measurement_value"]
        affinity_train_df["is_affinity"] = True
    else:
        affinity_train_df = None

    (affinity_model,) = AFFINITY_PREDICTOR.class1_pan_allele_models
    presentation_model = Class1PresentationNeuralNetwork(
        auxiliary_input_features=["gene"],
        batch_generator_batch_size=1024,
        max_epochs=10,
        learning_rate=0.001,
        patience=5,
        min_delta=0.0,
        random_negative_rate=1.0,
        random_negative_constant=25)
    presentation_model.load_from_class1_neural_network(affinity_model)

    presentation_predictor = Class1PresentationPredictor(
        models=[presentation_model],
        allele_to_sequence=AFFINITY_PREDICTOR.allele_to_sequence)

    mms_allele_encoding = MultipleAlleleEncoding(
        experiment_names=["experiment1"] * len(mms_train_df),
        experiment_to_allele_list={
            "experiment1": alleles,
        }, max_alleles_per_experiment=6,
        allele_to_sequence=AFFINITY_PREDICTOR.allele_to_sequence, )
    allele_encoding = copy.deepcopy(mms_allele_encoding)
    if affinity_train_df is not None:
        allele_encoding.append_alleles(affinity_train_df.allele.values)
        train_df = pandas.concat([mms_train_df, affinity_train_df],
            ignore_index=True, sort=False)
    else:
        train_df = mms_train_df

    allele_encoding = allele_encoding.compact()
    mms_allele_encoding = mms_allele_encoding.compact()

    pre_predictions = presentation_model.predict(
        peptides=mms_train_df.peptide.values,
        allele_encoding=mms_allele_encoding).score

    expected_pre_predictions = from_ic50(affinity_model.predict(
        peptides=numpy.repeat(mms_train_df.peptide.values, len(alleles)),
        allele_encoding=mms_allele_encoding.allele_encoding, )).reshape(
        (-1, len(alleles)))
    assert_allclose(pre_predictions, expected_pre_predictions, rtol=1e-4)

    random_peptides_encodable = EncodableSequences.create(
        random_peptides(10000, 9))

    original_motif = make_motif(
        presentation_predictor=presentation_predictor,
        peptides=random_peptides_encodable,
        allele=refine_allele)
    print("Original motif proline-3 rate: ", original_motif.loc[3, "P"])
    assert_less(original_motif.loc[3, "P"], 0.1)

    iteration_box = [0]

    def progress(label = None):
        if label is None:
            label = str(iteration_box[0])
            iteration_box[0] += 1
        print("*** iteration ", label, "***")
        predictions_df = presentation_predictor.predict_to_dataframe(
            peptides=mms_train_df.peptide.values,
            alleles=mms_allele_encoding)
        merged_df = pandas.merge(mms_train_df, predictions_df, on="peptide")
        merged_hit_df = merged_df.loc[merged_df.hit == 1.0]
        correct_allele_fraction =  (
            merged_hit_df.allele == merged_hit_df.true_allele).mean()
        print("Correct allele fraction", correct_allele_fraction)
        print(
            "Mean score/affinity for hit",
            merged_df.loc[merged_df.hit == 1.0].score.mean(),
            merged_df.loc[merged_df.hit == 1.0].affinity.mean())
        print(
            "Mean score/affinity for decoy",
            merged_df.loc[merged_df.hit == 0.0].score.mean(),
            merged_df.loc[merged_df.hit == 0.0].affinity.mean())
        auc = roc_auc_score(merged_df.hit.values, merged_df.score.values)
        print("AUC", auc)
        return (auc, correct_allele_fraction)

    (pre_auc, pre_correct_allele_fraction) = progress(label="Pre fitting")
    presentation_model.fit(peptides=train_df.peptide.values,
        labels=train_df.label.values,
        inequalities=train_df.measurement_inequality.values,
        affinities_mask=train_df.is_affinity.values,
        allele_encoding=allele_encoding,
        progress_callback=progress)
    (post_auc, post_correct_allele_fraction) = progress(label="Done fitting")

    final_motif = make_motif(
        presentation_predictor=presentation_predictor,
        peptides=random_peptides_encodable,
        allele=refine_allele)
    print("Final motif proline-3 rate: ", final_motif.loc[3, "P"])

    assert_greater(post_auc, pre_auc)
    assert_greater(
        post_correct_allele_fraction, pre_correct_allele_fraction - 0.05)
    assert_greater(final_motif.loc[3, "P"], original_motif.loc[3, "P"])



def test_synthetic_allele_refinement(include_affinities=True):
    """
    Test that in a synthetic example the model is able to learn that HLA-C*01:02
    prefers P at position 3.
    """
    refine_allele = "HLA-C*01:02"
    alleles = ["HLA-A*02:01", "HLA-B*27:01", "HLA-C*07:01", "HLA-A*03:01",
        "HLA-B*15:01", refine_allele]
    peptides_per_allele = [2000, 1000, 500, 1500, 1200, 800, ]

    allele_to_peptides = dict(zip(alleles, peptides_per_allele))

    length = 9

    train_with_ms = pandas.read_csv(get_path("data_curated",
        "curated_training_data.with_mass_spec.csv.bz2"))
    train_no_ms = pandas.read_csv(
        get_path("data_curated", "curated_training_data.no_mass_spec.csv.bz2"))

    def filter_df(df):
        return df.loc[
            (df.allele.isin(alleles)) & (df.peptide.str.len() == length)]

    train_with_ms = filter_df(train_with_ms)
    train_no_ms = filter_df(train_no_ms)

    ms_specific = train_with_ms.loc[
        ~train_with_ms.peptide.isin(train_no_ms.peptide)]

    train_peptides = []
    train_true_alleles = []
    for allele in alleles:
        peptides = ms_specific.loc[ms_specific.allele == allele].peptide.sample(
            n=allele_to_peptides[allele])
        train_peptides.extend(peptides)
        train_true_alleles.extend([allele] * len(peptides))

    hits_df = pandas.DataFrame({"peptide": train_peptides})
    hits_df["true_allele"] = train_true_alleles
    hits_df["hit"] = 1.0

    decoys_df = hits_df.copy()
    decoys_df["peptide"] = decoys_df.peptide.map(scramble_peptide)
    decoys_df["true_allele"] = ""
    decoys_df["hit"] = 0.0

    mms_train_df = pandas.concat([hits_df, decoys_df], ignore_index=True)
    mms_train_df["label"] = mms_train_df.hit
    mms_train_df["is_affinity"] = False
    mms_train_df["measurement_inequality"] = None

    if include_affinities:
        affinity_train_df = pandas.read_csv(get_path("models_class1_pan",
            "models.with_mass_spec/train_data.csv.bz2"))
        affinity_train_df = affinity_train_df.loc[
            affinity_train_df.allele.isin(alleles), ["peptide", "allele",
                "measurement_inequality", "measurement_value"]]

        affinity_train_df["label"] = affinity_train_df["measurement_value"]
        del affinity_train_df["measurement_value"]
        affinity_train_df["is_affinity"] = True
    else:
        affinity_train_df = None

    (affinity_model,) = AFFINITY_PREDICTOR.class1_pan_allele_models
    presentation_model = Class1PresentationNeuralNetwork(
        #batch_generator="multiallelic_mass_spec",
        batch_generator="simple",
        auxiliary_input_features=["gene"],
        batch_generator_batch_size=1024,
        max_epochs=10,
        learning_rate=0.0001,
        patience=5,
        min_delta=0.0,
        random_negative_rate=0,
        random_negative_constant=0)
    presentation_model.load_from_class1_neural_network(affinity_model)

    presentation_predictor = Class1PresentationPredictor(
        models=[presentation_model],
        allele_to_sequence=AFFINITY_PREDICTOR.allele_to_sequence)

    mms_allele_encoding = MultipleAlleleEncoding(
        experiment_names=["experiment1"] * len(mms_train_df),
        experiment_to_allele_list={
            "experiment1": alleles,
        }, max_alleles_per_experiment=6,
        allele_to_sequence=AFFINITY_PREDICTOR.allele_to_sequence)
    allele_encoding = copy.deepcopy(mms_allele_encoding)
    if affinity_train_df is not None:
        allele_encoding.append_alleles(affinity_train_df.allele.values)
        train_df = pandas.concat([mms_train_df, affinity_train_df],
            ignore_index=True, sort=False)
    else:
        train_df = mms_train_df

    allele_encoding = allele_encoding.compact()
    mms_allele_encoding = mms_allele_encoding.compact()

    pre_predictions = presentation_model.predict(
        peptides=mms_train_df.peptide.values,
        allele_encoding=mms_allele_encoding).score

    expected_pre_predictions = from_ic50(affinity_model.predict(
        peptides=numpy.repeat(mms_train_df.peptide.values, len(alleles)),
        allele_encoding=mms_allele_encoding.allele_encoding, )).reshape(
        (-1, len(alleles)))
    assert_allclose(pre_predictions, expected_pre_predictions, rtol=1e-4)

    random_peptides_encodable = EncodableSequences.create(
        random_peptides(10000, 9))

    original_motif = make_motif(
        presentation_predictor=presentation_predictor,
        peptides=random_peptides_encodable,
        allele=refine_allele)
    print("Original motif proline-3 rate: ", original_motif.loc[3, "P"])
    assert_less(original_motif.loc[3, "P"], 0.1)

    iteration_box = [0]

    def progress(label = None):
        if label is None:
            label = str(iteration_box[0])
            iteration_box[0] += 1
        print("*** iteration ", label, "***")
        predictions_df = presentation_predictor.predict_to_dataframe(
            peptides=mms_train_df.peptide.values,
            alleles=mms_allele_encoding)
        merged_df = pandas.merge(mms_train_df, predictions_df, on="peptide")
        merged_hit_df = merged_df.loc[merged_df.hit == 1.0]
        correct_allele_fraction =  (
            merged_hit_df.allele == merged_hit_df.true_allele).mean()
        print("Correct allele fraction", correct_allele_fraction)
        print(
            "Mean score/affinity for hit",
            merged_df.loc[merged_df.hit == 1.0].score.mean(),
            merged_df.loc[merged_df.hit == 1.0].affinity.mean())
        print(
            "Mean score/affinity for decoy",
            merged_df.loc[merged_df.hit == 0.0].score.mean(),
            merged_df.loc[merged_df.hit == 0.0].affinity.mean())
        auc = roc_auc_score(merged_df.hit.values, merged_df.score.values)
        print("AUC", auc)

        motif = make_motif(
            presentation_predictor=presentation_predictor,
            peptides=random_peptides_encodable,
            allele=refine_allele,
            master_allele_encoding=allele_encoding.allele_encoding)
        print("Proline-3 rate: ", motif.loc[3, "P"])

        return (auc, correct_allele_fraction)

    (pre_auc, pre_correct_allele_fraction) = progress(label="Pre fitting")
    presentation_model.fit(peptides=train_df.peptide.values,
        labels=train_df.label.values,
        inequalities=train_df.measurement_inequality.values,
        affinities_mask=train_df.is_affinity.values,
        allele_encoding=allele_encoding,
        progress_callback=progress)
    (post_auc, post_correct_allele_fraction) = progress(label="Done fitting")

    final_motif = make_motif(
        presentation_predictor=presentation_predictor,
        peptides=random_peptides_encodable,
        allele=refine_allele)
    print("Final motif proline-3 rate: ", final_motif.loc[3, "P"])

    assert_greater(post_auc, pre_auc)
    assert_greater(
        post_correct_allele_fraction, pre_correct_allele_fraction - 0.05)
    assert_greater(final_motif.loc[3, "P"], original_motif.loc[3, "P"])


def Xtest_real_data_multiallelic_refinement(max_epochs=10):
    """
    Test on real data that we can learn that HLA-A*02:20 has a preference K at
    position 1.
    """
    (affinity_model,) = AFFINITY_PREDICTOR.class1_pan_allele_models
    presentation_model = Class1PresentationNeuralNetwork(
        auxiliary_input_features=["gene"],
        batch_generator_batch_size=1024,
        max_epochs=max_epochs,
        learning_rate=0.001,
        patience=5,
        min_delta=0.0,
        random_negative_rate=0,
        random_negative_constant=0)
    presentation_model.load_from_class1_neural_network(affinity_model)

    presentation_predictor = Class1PresentationPredictor(
        models=[presentation_model],
        allele_to_sequence=AFFINITY_PREDICTOR.allele_to_sequence)

    ms_df = pandas.read_csv(
        get_path("data_mass_spec_annotated", "annotated_ms.csv.bz2"))
    ms_df = ms_df.loc[
        (ms_df.mhc_class == "I") & (~ms_df.protein_ensembl.isnull())].copy()

    sample_table = ms_df.drop_duplicates(
        "sample_id").set_index("sample_id").loc[ms_df.sample_id.unique()]
    grouped = ms_df.groupby("sample_id").nunique()
    for col in sample_table.columns:
        if (grouped[col] > 1).any():
            del sample_table[col]
    sample_table["alleles"] = sample_table.hla.str.split()

    multi_train_hit_df = ms_df.loc[
        ms_df.sample_id  == "RA957"
    ].drop_duplicates("peptide")[["peptide", "sample_id"]].reset_index(drop=True)
    multi_train_hit_df["label"] = 1.0

    multi_train_decoy_df = ms_df.loc[
        (ms_df.sample_id  == "CD165") &
        (~ms_df.peptide.isin(multi_train_hit_df.peptide.unique()))
    ].drop_duplicates("peptide")[["peptide"]]
    (multi_train_decoy_df["sample_id"],) = multi_train_hit_df.sample_id.unique()
    multi_train_decoy_df["label"] = 0.0

    multi_train_df = pandas.concat(
        [multi_train_hit_df, multi_train_decoy_df], ignore_index=True)
    multi_train_df["is_affinity"] = False

    multi_train_alleles = set()
    for alleles in sample_table.loc[multi_train_df.sample_id.unique()].alleles:
        multi_train_alleles.update(alleles)
    multi_train_alleles = sorted(multi_train_alleles)

    pan_train_df = pandas.read_csv(
        get_path(
            "models_class1_pan", "models.with_mass_spec/train_data.csv.bz2"))

    pan_sub_train_df = pan_train_df.loc[
        #pan_train_df.allele.isin(multi_train_alleles),
        :,
        ["peptide", "allele", "measurement_inequality", "measurement_value"]
    ]
    pan_sub_train_df["label"] = pan_sub_train_df["measurement_value"]
    del pan_sub_train_df["measurement_value"]
    pan_sub_train_df["is_affinity"] = True

    allele_encoding = MultipleAlleleEncoding(
        experiment_names=multi_train_df.sample_id.values,
        experiment_to_allele_list=sample_table.alleles.to_dict(),
        max_alleles_per_experiment=sample_table.alleles.str.len().max(),
        allele_to_sequence=AFFINITY_PREDICTOR.allele_to_sequence,
    )
    allele_encoding.append_alleles(pan_sub_train_df.allele.values)
    allele_encoding =  allele_encoding.compact()

    combined_train_df = pandas.concat([multi_train_df, pan_sub_train_df])

    refine_allele = "HLA-A*02:20"
    random_peptides_encodable = EncodableSequences.create(
        random_peptides(10000, 9))

    original_motif = make_motif(
        presentation_predictor=presentation_predictor,
        peptides=random_peptides_encodable,
        allele=refine_allele)
    print(
        refine_allele,
        "original motif lysine-1 rate: ",
        original_motif.loc[1, "K"])

    def progress():
        motif = make_motif(
            presentation_predictor=presentation_predictor,
            peptides=random_peptides_encodable,
            allele=refine_allele,
            master_allele_encoding=allele_encoding.allele_encoding)
        print(
            refine_allele,
            "current motif lysine-1 rate: ",
            motif.loc[1, "K"])

    presentation_model.fit(
        peptides=combined_train_df.peptide.values,
        labels=combined_train_df.label.values,
        allele_encoding=allele_encoding,
        affinities_mask=combined_train_df.is_affinity.values,
        inequalities=combined_train_df.measurement_inequality.values,
        progress_callback=progress)

    final_motif = make_motif(
        presentation_predictor=presentation_predictor,
        peptides=random_peptides_encodable,
        allele=refine_allele)
    print(refine_allele, "final motif lysine-1 rate: ", final_motif.loc[1, "K"])

    assert_greater(final_motif.loc[1, "K"], original_motif.loc[1, "K"])

if __name__ == "__main__":
    setup()
    test_real_data_multiallelic_refinement()
    teardown()