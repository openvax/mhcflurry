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
from functools import partial

from numpy.testing import assert_, assert_equal, assert_allclose
import numpy
from random import shuffle

from sklearn.metrics import roc_auc_score

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork
from mhcflurry.allele_encoding import MultipleAlleleEncoding
from mhcflurry.class1_ligandome_predictor import Class1LigandomePredictor
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.downloads import get_path
from mhcflurry.regression_target import from_ic50
from mhcflurry.common import random_peptides, positional_frequency_matrix
from mhcflurry.testing_utils import cleanup, startup
from mhcflurry.amino_acid import COMMON_AMINO_ACIDS

COMMON_AMINO_ACIDS = sorted(COMMON_AMINO_ACIDS)

PAN_ALLELE_PREDICTOR_NO_MASS_SPEC = None
PAN_ALLELE_MOTIFS_WITH_MASS_SPEC_DF = None
PAN_ALLELE_MOTIFS_NO_MASS_SPEC_DF = None


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
        for alpha in [None, 1.0, 20.0]:
            print("delta", delta)
            print("alpha", alpha)
            # Hit labels
            y_true = [
                1.0,
                0.0,
                1.0,
                1.0,
                0.0
            ]
            y_true = numpy.array(y_true)
            y_pred = [
                [0.3, 0.7, 0.5],
                [0.2, 0.4, 0.6],
                [0.1, 0.5, 0.3],
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

            if alpha is None:
                max_func = max
            else:
                max_func = partial(smooth_max, alpha=alpha)

            contributions = []
            for i in range(len(y_true)):
                if y_true[i] == 1.0:
                    for j in range(len(y_true)):
                        if y_true[j] == 0.0:
                            tightest_i = max_func(y_pred[i])
                            contribution = sum(
                                max(0, y_pred[j, k] - tightest_i + delta)**2
                                for k in range(y_pred.shape[1])
                            )
                            contributions.append(contribution)
            contributions = numpy.array(contributions)
            expected1 = contributions.sum()

            # reference implementation 2: numpy
            pos = numpy.array([
                max_func(y_pred[i])
                for i in range(len(y_pred))
                if y_true[i] == 1.0
            ])

            neg = y_pred[~y_true.astype(bool)]
            expected2 = (
                    numpy.maximum(0, neg.reshape((-1, 1)) - pos + delta)**2).sum()

            yield numpy.testing.assert_almost_equal, expected1, expected2, 4

            computed = evaluate_loss(
                partial(Class1LigandomePredictor.loss, delta=delta, alpha=alpha),
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

    predictor = Class1LigandomePredictor(
        PAN_ALLELE_PREDICTOR_NO_MASS_SPEC,
        max_ensemble_size=1,
        max_epochs=max_epochs,
        learning_rate=0.0001,
        patience=5,
        min_delta=0.0)

    allele_encoding = MultipleAlleleEncoding(
        experiment_names=["experiment1"] * len(train_df),
        experiment_to_allele_list={
            "experiment1": alleles,
        },
        max_alleles_per_experiment=6,
        allele_to_sequence=PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.allele_to_sequence,
    ).compact()

    pre_predictions = predictor.predict(
        peptides=train_df.peptide.values,
        allele_encoding=allele_encoding)

    (model,) = PAN_ALLELE_PREDICTOR_NO_MASS_SPEC.class1_pan_allele_models
    expected_pre_predictions = from_ic50(
        model.predict(
            peptides=numpy.repeat(train_df.peptide.values, len(alleles)),
            allele_encoding=allele_encoding.allele_encoding,
    )).reshape((-1, len(alleles)))

    train_df["pre_max_prediction"] = pre_predictions.max(1)
    pre_auc = roc_auc_score(train_df.hit.values, train_df.pre_max_prediction.values)
    print("PRE_AUC", pre_auc)

    assert_allclose(pre_predictions, expected_pre_predictions)

    motifs_history = []
    random_peptides_encodable = make_random_peptides(10000, [9])


    def update_motifs():
        for allele in alleles:
            motif = make_motif(allele, random_peptides_encodable)
            motifs_history.append((allele, motif))

    metric_rows = []

    def progress():
        predictions = predictor.predict(peptides=train_df.peptide.values,
            allele_encoding=allele_encoding, )

        train_df["max_prediction"] = predictions.max(1)
        train_df["predicted_allele"] = pandas.Series(alleles).loc[
            predictions.argmax(1).flatten()].values

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

        print("Mean prediction for hit", mean_predictions_for_hit)
        print("Mean prediction for decoy", mean_predictions_for_decoy)
        print("Correct predicted allele fraction", correct_allele_fraction)
        print("AUC", auc)

        metric_rows.append((
            mean_predictions_for_hit,
            mean_predictions_for_decoy,
            correct_allele_fraction,
            auc,
        ))

        update_motifs()

        return (predictions, auc)

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
            "mean_predictions_for_hit",
            "mean_predictions_for_decoy",
            "correct_allele_fraction",
            "auc"
        ])

    #import ipdb ; ipdb.set_trace()

    return (predictor, predictions, metrics, motifs)


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
