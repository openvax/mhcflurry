import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

import pandas
import tempfile
import pickle

from numpy.testing import assert_, assert_equal, assert_allclose, assert_array_equal
from nose.tools import assert_greater, assert_less
import numpy

from sklearn.metrics import roc_auc_score

from mhcflurry import Class1AffinityPredictor, Class1ProcessingPredictor
from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor
from mhcflurry.downloads import get_path
from mhcflurry.testing_utils import cleanup, startup
import mhcflurry.class1_presentation_predictor
mhcflurry.class1_presentation_predictor.PREDICT_CHUNK_SIZE = 15

from . import data_path

AFFINITY_PREDICTOR = None
CLEAVAGE_PREDICTOR = None
CLEAVAGE_PREDICTOR_NO_FLANKING = None
PRESENTATION_PREDICTOR = None


def setup():
    global AFFINITY_PREDICTOR
    global CLEAVAGE_PREDICTOR
    global CLEAVAGE_PREDICTOR_NO_FLANKING
    global PRESENTATION_PREDICTOR
    startup()
    AFFINITY_PREDICTOR = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"),
        optimization_level=0,
        max_models=1)
    CLEAVAGE_PREDICTOR = Class1ProcessingPredictor.load(
        get_path("models_class1_processing", "models"), max_models=1)
    CLEAVAGE_PREDICTOR_NO_FLANKING = Class1ProcessingPredictor.load(
        get_path("models_class1_processing_variants", "models.selected.no_flank"),
        max_models=1)
    PRESENTATION_PREDICTOR = Class1PresentationPredictor.load()


def teardown():
    global AFFINITY_PREDICTOR
    global CLEAVAGE_PREDICTOR
    global CLEAVAGE_PREDICTOR_NO_FLANKING
    global PRESENTATION_PREDICTOR
    AFFINITY_PREDICTOR = None
    CLEAVAGE_PREDICTOR = None
    CLEAVAGE_PREDICTOR_NO_FLANKING = None
    PRESENTATION_PREDICTOR = None
    cleanup()


def test_basic():
    df = pandas.read_csv(data_path("multiallelic.benchmark.small.csv.bz2"))
    train_df = df.loc[
        df.sample_id.isin(sorted(df.sample_id.unique())[:3])
    ]
    test_df = df.loc[
        ~df.sample_id.isin(train_df.sample_id.unique())
    ]
    test_df = test_df.sample(frac=0.01, weights=test_df.hit + 0.01)

    experiment_to_alleles = (
        df.drop_duplicates("sample_id").set_index("sample_id").hla.str.split().to_dict())

    predictor = Class1PresentationPredictor(
        affinity_predictor=AFFINITY_PREDICTOR,
        processing_predictor_without_flanks=CLEAVAGE_PREDICTOR_NO_FLANKING,
        processing_predictor_with_flanks=CLEAVAGE_PREDICTOR)

    predictor.fit(
        targets=train_df.hit.values,
        peptides=train_df.peptide.values,
        sample_names=train_df.sample_id.values,
        alleles=experiment_to_alleles,
        n_flanks=train_df.n_flank.values,
        c_flanks=train_df.c_flank.values,
        verbose=2)

    def add_prediction_cols(test_df, predictor):
        test_df["prediction1"] = predictor.predict(
            peptides=test_df.peptide.values,
            sample_names=test_df.sample_id.values,
            alleles=experiment_to_alleles,
            n_flanks=test_df.n_flank.values,
            c_flanks=test_df.c_flank.values,
            verbose=2).presentation_score.values

        test_df["prediction2"] = predictor.predict(
            peptides=test_df.peptide.values,
            sample_names=test_df.sample_id.values,
            alleles=experiment_to_alleles,
            verbose=2).presentation_score.values

    add_prediction_cols(test_df, predictor)

    score1 = roc_auc_score(test_df.hit.values, test_df.prediction1.values)
    score2 = roc_auc_score(test_df.hit.values, test_df.prediction2.values)

    print("AUC", score1, score2)

    assert_greater(score1, 0.8)
    assert_greater(score2, 0.8)

    # Test saving, loading, pickling
    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    predictor.save(models_dir)
    predictor2 = Class1PresentationPredictor.load(models_dir)
    predictor3 = pickle.loads(
        pickle.dumps(predictor, protocol=pickle.HIGHEST_PROTOCOL))
    predictor4 = pickle.loads(
        pickle.dumps(predictor2, protocol=pickle.HIGHEST_PROTOCOL))

    for (i, other_predictor) in enumerate([predictor2, predictor3, predictor4]):
        print("Testing identity", i + 1)
        other_test_df = test_df.copy()
        del other_test_df["prediction1"]
        del other_test_df["prediction2"]
        add_prediction_cols(other_test_df, other_predictor)
        numpy.testing.assert_array_almost_equal(
            test_df["prediction1"], other_test_df["prediction1"], decimal=6)
        numpy.testing.assert_array_almost_equal(
            test_df["prediction2"], other_test_df["prediction2"], decimal=6)


def test_downloaded_predictor_small():
    global PRESENTATION_PREDICTOR

    # Test sequence scanning
    scan_results = PRESENTATION_PREDICTOR.predict_sequences(
        sequences=[
            "MESLVPGFN",
            "QPYVFIKRS",
            "AGGHSYGAD",
        ],
        alleles={
            "HLA-A*02:01": ["HLA-A*02:01"],
            "HLA-C*02:01": ["HLA-C*02:01"],
        },
        peptide_lengths=[9],
        result="best")
    print(scan_results)
    assert_equal(len(scan_results), 6)

    scan_results = PRESENTATION_PREDICTOR.predict_sequences(
        sequences=[
            "MESLVPGFN",
            "QPYVFIKRS",
            "AGGHSYGAD",
        ],
        alleles={
            "HLA-A*02:01": ["HLA-A*02:01"],
            "HLA-C*02:01": ["HLA-C*02:01"],
        },
        peptide_lengths=[8, 9],
        result="best")
    print(scan_results)
    assert_equal(len(scan_results), 6)

    scan_results = PRESENTATION_PREDICTOR.predict_sequences(
        sequences=[
            "MESLVPGFN",
            "QPYVFIKRS",
            "AGGHSYGAD",
        ],
        alleles={
            "HLA-A*02:01": ["HLA-A*02:01"],
            "HLA-C*02:01": ["HLA-C*02:01"],
        },
        peptide_lengths=[9],
        result="all")
    print(scan_results)
    assert_equal(len(scan_results), 6)

    scan_results = PRESENTATION_PREDICTOR.predict_sequences(
        sequences=[
            "MESLVPGFN",
            "QPYVFIKRS",
            "AGGHSYGAD",
        ],
        alleles={
            "HLA-A*02:01": ["HLA-A*02:01"],
            "HLA-C*02:01": ["HLA-C*02:01"],
        },
        peptide_lengths=[8, 9],
        result="all")
    print(scan_results)
    assert_equal(len(scan_results), 18)

    scan_results = PRESENTATION_PREDICTOR.predict_sequences(
        sequences=[
            "MESLVPGFN",
            "QPYVFIKRS",
            "AGGHSYGAD",
        ],
        alleles={
            "HLA-A*02:01": ["HLA-A*02:01"],
            "HLA-C*02:01": ["HLA-C*02:01"],
        },
        peptide_lengths=[10],
        result="all")
    print(scan_results)
    assert_equal(len(scan_results), 0)


def test_downloaded_predictor():
    global PRESENTATION_PREDICTOR

    # Test sequence scanning
    scan_results1 = PRESENTATION_PREDICTOR.predict_sequences(
        sequences=[
            "MESLVPGFNEKTHVQLSLPVLQVRDVLVRGFGDSVEEVLSEARQHLKDGTCGLVEVEKGVLPQLE",
            "QPYVFIKRSDARTAPHGHVMVELVAELEGIQYGRSGETLGVLVPHVGEIPVAYRKVLLRKNGNKG",
            "AGGHSYGADLKSFDLGDELGTDPYEDFQENWNTKHSSGVTRELMRELNGGAYTRYVDNNFCGPDG",
        ],
        alleles=[
            "HLA-A*02:01",
            "HLA-A*03:01",
            "HLA-B*57:01",
            "HLA-B*44:02",
            "HLA-C*02:01",
            "HLA-C*07:01",
        ])
    print(scan_results1)

    assert_equal(len(scan_results1), 3)
    assert (scan_results1.affinity < 200).all()
    assert (scan_results1.presentation_score > 0.7).all()

    scan_results2 = PRESENTATION_PREDICTOR.predict_sequences(
        result="filtered",
        filter_value=500,
        comparison_quantity="affinity",
        sequences={
            "seq1": "MESLVPGFNEKTHVQLSLPVLQVRDVLVRGFGDSVEEVLSEARQHLKDGTCGLVEVEKGVLPQLE",
            "seq2": "QPYVFIKRSDARTAPHGHVMVELVAELEGIQYGRSGETLGVLVPHVGEIPVAYRKVLLRKNGNKG",
            "seq3": "AGGHSYGADLKSFDLGDELGTDPYEDFQENWNTKHSSGVTRELMRELNGGAYTRYVDNNFCGPDG",
        },
        alleles=[
            "HLA-A*02:01",
            "HLA-A*03:01",
            "HLA-B*57:01",
            "HLA-B*44:02",
            "HLA-C*02:01",
            "HLA-C*07:01",
        ])
    print(scan_results2)

    assert len(scan_results2) > 10
    assert (scan_results2.affinity <= 500).all()

    scan_results3 = PRESENTATION_PREDICTOR.predict_sequences(
        result="filtered",
        filter_value=0.9,
        comparison_quantity="presentation_score",
        sequences={
            "seq1": "MESLVPGFNEKTHVQLSLPVLQVRDVLVRGFGDSVEEVLSEARQHLKDGTCGLVEVEKGVLPQLE",
            "seq2": "QPYVFIKRSDARTAPHGHVMVELVAELEGIQYGRSGETLGVLVPHVGEIPVAYRKVLLRKNGNKG",
            "seq3": "AGGHSYGADLKSFDLGDELGTDPYEDFQENWNTKHSSGVTRELMRELNGGAYTRYVDNNFCGPDG",
        },
        alleles=[
            "HLA-A*02:01",
            "HLA-A*03:01",
            "HLA-B*57:01",
            "HLA-B*44:02",
            "HLA-C*02:01",
            "HLA-C*07:01",
        ])
    print(scan_results3)

    assert len(scan_results3) > 5, len(scan_results3)
    assert (scan_results3.presentation_score >= 0.9).all()

    scan_results4 = PRESENTATION_PREDICTOR.predict_sequences(
        result="all",
        comparison_quantity="affinity",
        sequences={
            "seq1": "MESLVPGFNEKTHVQLSLPVLQVRDVLVRGFGDSVEEVLSEARQHLKDGTCGLVEVEKGVLPQLE",
            "seq2": "QPYVFIKRSDARTAPHGHVMVELVAELEGIQYGRSGETLGVLVPHVGEIPVAYRKVLLRKNGNKG",
            "seq3": "AGGHSYGADLKSFDLGDELGTDPYEDFQENWNTKHSSGVTRELMRELNGGAYTRYVDNNFCGPDG",
        },
        alleles=[
            "HLA-A*02:01",
            "HLA-A*03:01",
            "HLA-B*57:01",
            "HLA-B*44:02",
            "HLA-C*02:01",
            "HLA-C*07:01",
        ])
    print(scan_results4)

    assert len(scan_results4) > 200, len(scan_results4)
    assert_less(scan_results4.iloc[0].affinity, 100)

    scan_results5 = PRESENTATION_PREDICTOR.predict_sequences(
        result="all",
        comparison_quantity="affinity",
        sequences={
            "seq1": "MESLVPGFNEKTHVQLSLPVLQVRDVLVRGFGDSVEEVLSEARQHLKDGTCGLVEVEKGVLPQLE",
            "seq2": "QPYVFIKRSDARTAPHGHVMVELVAELEGIQYGRSGETLGVLVPHVGEIPVAYRKVLLRKNGNKG",
            "seq3": "AGGHSYGADLKSFDLGDELGTDPYEDFQENWNTKHSSGVTRELMRELNGGAYTRYVDNNFCGPDG",
        },
        alleles={
            "sample1": [
                "HLA-A*02:01",
                "HLA-A*03:01",
                "HLA-B*57:01",
                "HLA-B*44:02",
                "HLA-C*02:01",
                "HLA-C*07:01",
            ],
            "sample2": [
                "HLA-A*01:01",
                "HLA-A*02:06",
                "HLA-B*07:02",
                "HLA-B*44:02",
                "HLA-C*03:01",
                "HLA-C*07:02",
            ],
        })
    print(scan_results5)
    assert_equal(len(scan_results5), len(scan_results4) * 2)
