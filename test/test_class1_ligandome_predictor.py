"""

Idea:

- take an allele where MS vs. no-MS trained predictors are very different. One
    possiblility is DLA-88*501:01 but human would be better
- generate synethetic multi-allele MS by combining single-allele MS for differnet
   alleles, including the selected allele
- train ligandome predictor based on the no-ms pan-allele models on theis
  synthetic dataset
- see if the pan-allele predictor learns the "correct" motif for the selected
  allele, i.e. updates to become more similar to the with-ms pan allele predictor.


"""

import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import pandas
import argparse
import sys
import copy
import os

from numpy.testing import assert_, assert_equal, assert_allclose
from nose.tools import assert_greater, assert_less
import numpy
from random import shuffle

from sklearn.metrics import roc_auc_score

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork
from mhcflurry.allele_encoding import MultipleAlleleEncoding
from mhcflurry.class1_ligandome_neural_network import Class1LigandomeNeuralNetwork
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.downloads import get_path
from mhcflurry.regression_target import from_ic50
from mhcflurry.common import random_peptides, positional_frequency_matrix
from mhcflurry.testing_utils import cleanup, startup
from mhcflurry.amino_acid import COMMON_AMINO_ACIDS
from mhcflurry.custom_loss import MultiallelicMassSpecLoss

COMMON_AMINO_ACIDS = sorted(COMMON_AMINO_ACIDS)

PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = None
PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF = None
PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF = None

def data_path(name):
    '''
    Return the absolute path to a file in the test/data directory.
    The name specified should be relative to test/data.
    '''
    return os.path.join(os.path.dirname(__file__), "data", name)



def setup():
    global PAN_ALLELE_PREDICTOR_NO_MASS_SPEC
    global PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF
    global PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF
    startup()
    PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.no_mass_spec"),
        optimization_level=0,
        max_models=1)

    PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF = pandas.read_csv(
        get_path(
            "models_class1_pan",
            "models.with_mass_spec/frequency_matrices.csv.bz2"))
    PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF = pandas.read_csv(
        get_path(
            "models_class1_pan",
            "models.no_mass_spec/frequency_matrices.csv.bz2"))


def teardown():
    global PAN_ALLELE_PREDICTOR_NO_MASS_SPEC
    global PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF
    global PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF

    PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = None
    PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF = None
    PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF = None
    cleanup()


def scramble_peptide(peptide):
    lst = list(peptide)
    shuffle(lst)
    return "".join(lst)


def evaluate_loss(loss, y_true, y_pred):
    import keras.backend as K

    y_true = numpy.array(y_true)
    y_pred = numpy.array(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((len(y_pred), 1))
    if y_true.ndim == 1:
        y_true = y_true.reshape((len(y_true), 1))

    if K.backend() == "tensorflow":
        session = K.get_session()
        y_true_var = K.constant(y_true, name="y_true")
        y_pred_var = K.constant(y_pred, name="y_pred")
        result = loss(y_true_var, y_pred_var)
        return result.eval(session=session)
    elif K.backend() == "theano":
        y_true_var = K.constant(y_true, name="y_true")
        y_pred_var = K.constant(y_pred, name="y_pred")
        result = loss(y_true_var, y_pred_var)
        return result.eval()
    else:
        raise ValueError("Unsupported backend: %s" % K.backend())


def test_loss():
    for delta in [0.0, 0.3]:
        print("delta", delta)
        # Hit labels
        y_true = [
            1.0,
            0.0,
            1.0,
            -1.0,  # ignored
            1.0,
            0.0
        ]
        y_true = numpy.array(y_true)
        y_pred = [
            [0.3, 0.7, 0.5],
            [0.2, 0.4, 0.6],
            [0.1, 0.5, 0.3],
            [0.9, 0.1, 0.2],
            [0.1, 0.7, 0.1],
            [0.8, 0.2, 0.4],

        ]
        y_pred = numpy.array(y_pred)

        # reference implementation 1

        def smooth_max(x, alpha):
            x = numpy.array(x)
            alpha = numpy.array([alpha])
            return (x * numpy.exp(x * alpha)).sum() / (
                numpy.exp(x * alpha)).sum()

        contributions = []
        for i in range(len(y_true)):
            if y_true[i] == 1.0:
                for j in range(len(y_true)):
                    if y_true[j] == 0.0:
                        tightest_i = max(y_pred[i])
                        contribution = sum(
                            max(0, y_pred[j, k] - tightest_i + delta)**2
                            for k in range(y_pred.shape[1])
                        )
                        contributions.append(contribution)
        contributions = numpy.array(contributions)
        expected1 = contributions.sum() / len(contributions)

        # reference implementation 2: numpy
        pos = numpy.array([
            max(y_pred[i])
            for i in range(len(y_pred))
            if y_true[i] == 1.0
        ])

        neg = y_pred[(y_true == 0.0).astype(bool)]
        expected2 = (
                numpy.maximum(0, neg.reshape((-1, 1)) - pos + delta)**2).sum() / (
            len(pos) * len(neg))

        yield numpy.testing.assert_almost_equal, expected1, expected2, 4

        computed = evaluate_loss(
            MultiallelicMassSpecLoss(delta=delta).loss,
            y_true,
            y_pred.reshape(y_pred.shape + (1,)))

        yield numpy.testing.assert_almost_equal, computed, expected1, 4


AA_DIST = pandas.Series(
    dict((line.split()[0], float(line.split()[1])) for line in """
A    0.071732
E    0.060102
N    0.034679
D    0.039601
T    0.055313
L    0.115337
V    0.070498
S    0.071882
Q    0.040436
F    0.050178
G    0.053176
C    0.005429
H    0.025487
I    0.056312
W    0.013593
K    0.057832
M    0.021079
Y    0.043372
R    0.060330
P    0.053632
""".strip().split("\n")))
print(AA_DIST)


def make_random_peptides(num_peptides_per_length=10000, lengths=[9]):
    peptides = []
    for length in lengths:
        peptides.extend(
            random_peptides
                (num_peptides_per_length, length=length, distribution=AA_DIST))
    return EncodableSequences.create(peptides)


def make_motif(allele, peptides, frac=0.01):
    peptides = EncodableSequences.create(peptides)
    predictions = PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.predict(
        peptides=peptides,
        allele=allele,
    )
    random_predictions_df = pandas.DataFrame({"peptide": peptides.sequences})
    random_predictions_df["prediction"] = predictions
    random_predictions_df = random_predictions_df.sort_values(
        "prediction", ascending=True)
    top = random_predictions_df.iloc[:int(len(random_predictions_df) * frac)]
    matrix = positional_frequency_matrix(top.peptide.values)
    return matrix


def test_real_data_multiallelic_refinement(max_epochs=10):
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
        pan_train_df.allele.isin(multi_train_alleles),
        ["peptide", "allele", "measurement_inequality", "measurement_value"]
    ]
    pan_sub_train_df["label"] = pan_sub_train_df["measurement_value"]
    del pan_sub_train_df["measurement_value"]
    pan_sub_train_df["is_affinity"] = True

    pan_predictor = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.with_mass_spec"),
        optimization_level=0,
        max_models=1)

    allele_encoding = MultipleAlleleEncoding(
        experiment_names=multi_train_df.sample_id.values,
        experiment_to_allele_list=sample_table.alleles.to_dict(),
        max_alleles_per_experiment=sample_table.alleles.str.len().max(),
        allele_to_sequence=pan_predictor.allele_to_sequence,
    )
    allele_encoding.append_alleles(pan_sub_train_df.allele.values)
    allele_encoding =  allele_encoding.compact()

    combined_train_df = pandas.concat([multi_train_df, pan_sub_train_df])

    ligandome_predictor = Class1LigandomeNeuralNetwork(
        pan_predictor,
        auxiliary_input_features=[],
        max_ensemble_size=1,
        max_epochs=max_epochs,
        learning_rate=0.0001,
        patience=5,
        min_delta=0.0,
        random_negative_rate=1.0)

    pre_predictions = from_ic50(
        ligandome_predictor.predict(
            output="affinities",
            peptides=combined_train_df.peptide.values,
            alleles=allele_encoding))

    (model,) = pan_predictor.class1_pan_allele_models
    expected_pre_predictions = from_ic50(
        model.predict(
            peptides=numpy.repeat(combined_train_df.peptide.values, len(alleles)),
            allele_encoding=allele_encoding.allele_encoding,
    )).reshape((-1, len(alleles)))[:,0]

    assert_allclose(pre_predictions, expected_pre_predictions, rtol=1e-4)

    motifs_history = []
    random_peptides_encodable = make_random_peptides(10000, [9])


    def update_motifs():
        for allele in multi_train_alleles:
            motif = make_motif(allele, random_peptides_encodable)
            motifs_history.append((allele, motif))

    print("Pre fitting:")
    update_motifs()
    print("Fitting...")

    ligandome_predictor.fit(
        peptides=combined_train_df.peptide.values,
        labels=combined_train_df.label.values,
        allele_encoding=allele_encoding,
        affinities_mask=combined_train_df.is_affinity.values,
        inequalities=combined_train_df.measurement_inequality.values,
        progress_callback=update_motifs,
    )

    import ipdb ; ipdb.set_trace()


def test_synthetic_allele_refinement_with_affinity_data(max_epochs=10):
    refine_allele = "HLA-C*01:02"
    alleles = [
        "HLA-A*02:01", "HLA-B*27:01", "HLA-C*07:01",
        "HLA-A*03:01", "HLA-B*15:01", refine_allele
    ]
    peptides_per_allele = [
        2000, 1000, 500,
        1500, 1200, 800,
    ]

    allele_to_peptides = dict(zip(alleles, peptides_per_allele))

    length = 9

    train_with_ms = pandas.read_csv(
        get_path("data_curated", "curated_training_data.with_mass_spec.csv.bz2"))
    train_no_ms = pandas.read_csv(get_path("data_curated",
        "curated_training_data.no_mass_spec.csv.bz2"))

    def filter_df(df):
        df = df.loc[
            (df.allele.isin(alleles)) &
            (df.peptide.str.len() == length)
        ]
        return df

    train_with_ms = filter_df(train_with_ms)
    train_no_ms = filter_df(train_no_ms)

    ms_specific = train_with_ms.loc[
        ~train_with_ms.peptide.isin(train_no_ms.peptide)
    ]

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
    mms_train_df["label"] =  mms_train_df.hit
    mms_train_df["is_affinity"] = False

    affinity_train_df = pandas.read_csv(
        get_path(
            "models_class1_pan", "models.with_mass_spec/train_data.csv.bz2"))
    affinity_train_df = affinity_train_df.loc[
        affinity_train_df.allele.isin(alleles),
        ["peptide", "allele",  "measurement_inequality", "measurement_value"]]

    affinity_train_df["label"] = affinity_train_df["measurement_value"]
    del affinity_train_df["measurement_value"]
    affinity_train_df["is_affinity"] = True

    predictor = Class1LigandomeNeuralNetwork(
        PAN_ALLELE_PREDICTOR_NO_MASS_SPEC,
        auxiliary_input_features=["gene"],
        max_ensemble_size=1,
        max_epochs=max_epochs,
        learning_rate=0.0001,
        patience=5,
        min_delta=0.0,
        random_negative_rate=1.0,
        random_negative_constant=25)

    mms_allele_encoding = MultipleAlleleEncoding(
        experiment_names=["experiment1"] * len(mms_train_df),
        experiment_to_allele_list={
            "experiment1": alleles,
        },
        max_alleles_per_experiment=6,
        allele_to_sequence=PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.allele_to_sequence,
    )
    allele_encoding = copy.deepcopy(mms_allele_encoding)
    allele_encoding.append_alleles(affinity_train_df.allele.values)
    allele_encoding = allele_encoding.compact()

    train_df = pandas.concat(
        [mms_train_df, affinity_train_df], ignore_index=True, sort=False)

    pre_predictions = from_ic50(
        predictor.predict(
            output="affinities_matrix",
            peptides=mms_train_df.peptide.values,
            alleles=mms_allele_encoding))

    (model,) = PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.class1_pan_allele_models
    expected_pre_predictions = from_ic50(
        model.predict(
            peptides=numpy.repeat(mms_train_df.peptide.values, len(alleles)),
            allele_encoding=mms_allele_encoding.allele_encoding,
    )).reshape((-1, len(alleles)))

    #import ipdb ; ipdb.set_trace()

    mms_train_df["pre_max_prediction"] = pre_predictions.max(1)
    pre_auc = roc_auc_score(mms_train_df.hit.values, mms_train_df.pre_max_prediction.values)
    print("PRE_AUC", pre_auc)

    assert_allclose(pre_predictions, expected_pre_predictions, rtol=1e-4)

    motifs_history = []
    random_peptides_encodable = make_random_peptides(10000, [9])


    def update_motifs():
        for allele in alleles:
            motif = make_motif(allele, random_peptides_encodable)
            motifs_history.append((allele, motif))

    metric_rows = []

    def progress():
        (_, ligandome_prediction, affinities_predictions) = (
            predictor.predict(
                output="all",
                peptides=mms_train_df.peptide.values,
                alleles=mms_allele_encoding))
        affinities_predictions = from_ic50(affinities_predictions)
        for (kind, predictions) in [
                ("affinities", affinities_predictions),
                ("ligandome", ligandome_prediction)]:

            mms_train_df["max_prediction"] = predictions.max(1)
            mms_train_df["predicted_allele"] = pandas.Series(alleles).loc[
                predictions.argmax(1).flatten()
            ].values

            print(kind)
            print(predictions)

            mean_predictions_for_hit = mms_train_df.loc[
                mms_train_df.hit == 1.0
            ].max_prediction.mean()
            mean_predictions_for_decoy = mms_train_df.loc[
                mms_train_df.hit == 0.0
            ].max_prediction.mean()
            correct_allele_fraction = (
                    mms_train_df.loc[mms_train_df.hit == 1.0].predicted_allele ==
                    mms_train_df.loc[mms_train_df.hit == 1.0].true_allele
            ).mean()
            auc = roc_auc_score(mms_train_df.hit.values, mms_train_df.max_prediction.values)

            print(kind, "Mean prediction for hit", mean_predictions_for_hit)
            print(kind, "Mean prediction for decoy", mean_predictions_for_decoy)
            print(kind, "Correct predicted allele fraction", correct_allele_fraction)
            print(kind, "AUC", auc)

            metric_rows.append((
                kind,
                mean_predictions_for_hit,
                mean_predictions_for_decoy,
                correct_allele_fraction,
                auc,
            ))

            update_motifs()

        return (ligandome_prediction, auc)

    print("Pre fitting:")
    progress()
    update_motifs()
    print("Fitting...")

    predictor.fit(
        peptides=train_df.peptide.values,
        labels=train_df.label.values,
        inequalities=train_df.measurement_inequality.values,
        affinities_mask=train_df.is_affinity.values,
        allele_encoding=allele_encoding,
        progress_callback=progress,
    )

    (predictions, final_auc) = progress()
    print("Final AUC", final_auc)

    update_motifs()

    motifs = pandas.DataFrame(
        motifs_history,
        columns=[
            "allele",
            "motif",
        ]
    )

    metrics = pandas.DataFrame(
        metric_rows,
        columns=[
            "output",
            "mean_predictions_for_hit",
            "mean_predictions_for_decoy",
            "correct_allele_fraction",
            "auc"
        ])

    return (predictor, predictions, metrics, motifs)



def test_synthetic_allele_refinement(max_epochs=10):
    refine_allele = "HLA-C*01:02"
    alleles = [
        "HLA-A*02:01", "HLA-B*27:01", "HLA-C*07:01",
        "HLA-A*03:01", "HLA-B*15:01", refine_allele
    ]
    peptides_per_allele = [
        2000, 1000, 500,
        1500, 1200, 800,
    ]

    allele_to_peptides = dict(zip(alleles, peptides_per_allele))

    length = 9

    train_with_ms = pandas.read_csv(
        get_path("data_curated", "curated_training_data.with_mass_spec.csv.bz2"))
    train_no_ms = pandas.read_csv(get_path("data_curated",
        "curated_training_data.no_mass_spec.csv.bz2"))

    def filter_df(df):
        df = df.loc[
            (df.allele.isin(alleles)) &
            (df.peptide.str.len() == length)
        ]
        return df

    train_with_ms = filter_df(train_with_ms)
    train_no_ms = filter_df(train_no_ms)

    ms_specific = train_with_ms.loc[
        ~train_with_ms.peptide.isin(train_no_ms.peptide)
    ]

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

    train_df = pandas.concat([hits_df, decoys_df], ignore_index=True)

    predictor = Class1LigandomeNeuralNetwork(
        PAN_ALLELE_PREDICTOR_NO_MASS_SPEC,
        max_ensemble_size=1,
        max_epochs=max_epochs,
        learning_rate=0.0001,
        patience=5,
        min_delta=0.0,
        random_negative_rate=0.0,
        random_negative_constant=0)

    allele_encoding = MultipleAlleleEncoding(
        experiment_names=["experiment1"] * len(train_df),
        experiment_to_allele_list={
            "experiment1": alleles,
        },
        max_alleles_per_experiment=6,
        allele_to_sequence=PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.allele_to_sequence,
    ).compact()

    pre_predictions = from_ic50(
        predictor.predict(
            output="affinities_matrix",
            peptides=train_df.peptide.values,
            alleles=allele_encoding))

    (model,) = PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.class1_pan_allele_models
    expected_pre_predictions = from_ic50(
        model.predict(
            peptides=numpy.repeat(train_df.peptide.values, len(alleles)),
            allele_encoding=allele_encoding.allele_encoding,
    )).reshape((-1, len(alleles)))

    #import ipdb ; ipdb.set_trace()

    train_df["pre_max_prediction"] = pre_predictions.max(1)
    pre_auc = roc_auc_score(train_df.hit.values, train_df.pre_max_prediction.values)
    print("PRE_AUC", pre_auc)

    assert_allclose(pre_predictions, expected_pre_predictions, rtol=1e-4)

    motifs_history = []
    random_peptides_encodable = make_random_peptides(10000, [9])


    def update_motifs():
        for allele in alleles:
            motif = make_motif(allele, random_peptides_encodable)
            motifs_history.append((allele, motif))

    metric_rows = []

    def progress():
        (_, ligandome_prediction, affinities_predictions) = (
            predictor.predict(
                output="all",
                peptides=train_df.peptide.values,
                alleles=allele_encoding))
        affinities_predictions = from_ic50(affinities_predictions)
        for (kind, predictions) in [
                ("affinities", affinities_predictions),
                ("ligandome", ligandome_prediction)]:

            train_df["max_prediction"] = predictions.max(1)
            train_df["predicted_allele"] = pandas.Series(alleles).loc[
                predictions.argmax(1).flatten()
            ].values

            print(kind)
            print(predictions)

            mean_predictions_for_hit = train_df.loc[
                train_df.hit == 1.0
            ].max_prediction.mean()
            mean_predictions_for_decoy = train_df.loc[
                train_df.hit == 0.0
            ].max_prediction.mean()
            correct_allele_fraction = (
                    train_df.loc[train_df.hit == 1.0].predicted_allele ==
                    train_df.loc[train_df.hit == 1.0].true_allele
            ).mean()
            auc = roc_auc_score(train_df.hit.values, train_df.max_prediction.values)

            print(kind, "Mean prediction for hit", mean_predictions_for_hit)
            print(kind, "Mean prediction for decoy", mean_predictions_for_decoy)
            print(kind, "Correct predicted allele fraction", correct_allele_fraction)
            print(kind, "AUC", auc)

            metric_rows.append((
                kind,
                mean_predictions_for_hit,
                mean_predictions_for_decoy,
                correct_allele_fraction,
                auc,
            ))

            update_motifs()

        return (ligandome_prediction, auc)

    print("Pre fitting:")
    progress()
    update_motifs()
    print("Fitting...")

    predictor.fit(
        peptides=train_df.peptide.values,
        labels=train_df.hit.values,
        allele_encoding=allele_encoding,
        progress_callback=progress,
    )

    (predictions, final_auc) = progress()
    print("Final AUC", final_auc)

    update_motifs()

    motifs = pandas.DataFrame(
        motifs_history,
        columns=[
            "allele",
            "motif",
        ]
    )

    metrics = pandas.DataFrame(
        metric_rows,
        columns=[
            "output",
            "mean_predictions_for_hit",
            "mean_predictions_for_decoy",
            "correct_allele_fraction",
            "auc"
        ])

    return (predictor, predictions, metrics, motifs)


def test_batch_generator(sample_rate=0.1):
    multi_train_df = pandas.read_csv(
        data_path("multiallelic_ms.benchmark1.csv.bz2"))
    multi_train_df["label"] = multi_train_df.hit
    multi_train_df["is_affinity"] = False

    sample_table = multi_train_df.loc[
        multi_train_df.label == True
    ].drop_duplicates("sample_id").set_index("sample_id").loc[
        multi_train_df.sample_id.unique()
    ]
    grouped = multi_train_df.groupby("sample_id").nunique()
    for col in sample_table.columns:
        if (grouped[col] > 1).any():
            del sample_table[col]
    sample_table["alleles"] = sample_table.hla.str.split()

    pan_train_df = pandas.read_csv(
        get_path(
            "models_class1_pan", "models.with_mass_spec/train_data.csv.bz2"))
    pan_sub_train_df = pan_train_df
    pan_sub_train_df["label"] = pan_sub_train_df["measurement_value"]
    del pan_sub_train_df["measurement_value"]
    pan_sub_train_df["is_affinity"] = True

    pan_sub_train_df = pan_sub_train_df.sample(frac=sample_rate)
    multi_train_df = multi_train_df.sample(frac=sample_rate)

    pan_predictor = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.with_mass_spec"),
        optimization_level=0,
        max_models=1)

    allele_encoding = MultipleAlleleEncoding(
        experiment_names=multi_train_df.sample_id.values,
        experiment_to_allele_list=sample_table.alleles.to_dict(),
        max_alleles_per_experiment=sample_table.alleles.str.len().max(),
        allele_to_sequence=pan_predictor.allele_to_sequence,
    )
    allele_encoding.append_alleles(pan_sub_train_df.allele.values)
    allele_encoding = allele_encoding.compact()

    combined_train_df = pandas.concat(
        [multi_train_df, pan_sub_train_df], ignore_index=True, sort=True)

    ligandome_predictor = Class1LigandomeNeuralNetwork(
        pan_predictor,
        auxiliary_input_features=[],
        max_ensemble_size=1,
        max_epochs=0,
        batch_generator_batch_size=128,
        learning_rate=0.0001,
        patience=5,
        min_delta=0.0,
        random_negative_rate=1.0)

    fit_results = ligandome_predictor.fit(
        peptides=combined_train_df.peptide.values,
        labels=combined_train_df.label.values,
        allele_encoding=allele_encoding,
        affinities_mask=combined_train_df.is_affinity.values,
        inequalities=combined_train_df.measurement_inequality.values,
    )

    batch_generator = fit_results['batch_generator']
    train_batch_plan = batch_generator.train_batch_plan

    assert_greater(len(train_batch_plan.equivalence_class_labels), 100)
    assert_less(len(train_batch_plan.equivalence_class_labels), 1000)


parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument(
    "--out-metrics-csv",
    default=None,
    help="Metrics output")
parser.add_argument(
    "--out-motifs-pickle",
    default=None,
    help="Metrics output")
parser.add_argument(
    "--max-epochs",
    default=100,
    type=int,
    help="Max epochs")




if __name__ == '__main__':
    # If run directly from python, leave the user in a shell to explore results.
    setup()
    args = parser.parse_args(sys.argv[1:])
    (predictor, predictions, metrics, motifs) = (
        test_synthetic_allele_refinement(max_epochs=args.max_epochs))

    if args.out_metrics_csv:
        metrics.to_csv(args.out_metrics_csv)
    if args.out_motifs_pickle:
        motifs.to_pickle(args.out_motifs_pickle)

    # Leave in ipython
    import ipdb  # pylint: disable=import-error
    ipdb.set_trace()
