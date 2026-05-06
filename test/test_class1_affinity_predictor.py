"""Tests for Class1AffinityPredictor."""
import pytest

import tempfile
import shutil
import logging
import shlex
import warnings
import traceback
import sys

import numpy
import pandas

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork

from numpy import testing

from mhcflurry.downloads import get_path
from mhcflurry.percent_rank_transform import PercentRankTransform
from mhcflurry.pseudosequences import (
    LEGACY_ALLELE_SEQUENCES_FILENAME,
    LEGACY_CLASS1_PSEUDOSEQUENCES_FILENAME,
    PSEUDOSEQUENCE_FILENAMES_BY_LENGTH,
)
from mhcflurry.testing_utils import cleanup, startup

DOWNLOADED_PREDICTOR = None


@pytest.fixture(autouse=True, scope="module")
def setup_teardown():
    """Load the downloaded predictor once for this module."""
    global DOWNLOADED_PREDICTOR
    startup()
    try:
        DOWNLOADED_PREDICTOR = Class1AffinityPredictor.load()
    except Exception:
        DOWNLOADED_PREDICTOR = None
    logging.basicConfig(level=logging.DEBUG)
    yield
    DOWNLOADED_PREDICTOR = None
    cleanup()


# To hunt down a weird warning we were seeing in pandas.
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def test_save_calibration_only_preserves_model_artifacts(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    info_path = tmp_path / "info.txt"
    allele_sequences_path = tmp_path / LEGACY_ALLELE_SEQUENCES_FILENAME
    optimization_info_path = tmp_path / "optimization_info.json"
    motif_summary_path = tmp_path / "motif_summary.csv.bz2"

    manifest_text = (
        "model_name,allele,config_json\n"
        "PAN-CLASS1-0,pan-class1,\"{}\"\n"
        "PAN-CLASS1-1,pan-class1,\"{}\"\n"
    )
    info_text = "trained on\toriginal release\n"
    allele_sequences_text = "allele,sequence\nHLA-A*02:01,ORIGINAL\n"
    manifest_path.write_text(manifest_text)
    info_path.write_text(info_text)
    allele_sequences_path.write_text(allele_sequences_text)

    transform = PercentRankTransform()
    transform.fit(numpy.array([10.0, 20.0, 30.0]), bins=3)
    predictor = Class1AffinityPredictor(
        allele_to_sequence={"HLA-A*02:01": "REWRITTEN"},
        allele_to_percent_rank_transform={"HLA-A*02:01": transform},
        metadata_dataframes={
            "motif_summary": pandas.DataFrame({
                "allele": ["HLA-A*02:01"],
                "value": [1.0],
            }),
        },
        optimization_info={
            "pan_models_merged": True,
            "num_pan_models_merged": 2,
        },
    )

    predictor.save(str(tmp_path), model_names_to_write=[])

    assert manifest_path.read_text() == manifest_text
    assert info_path.read_text() == info_text
    assert allele_sequences_path.read_text() == allele_sequences_text
    assert not (tmp_path / PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[37]).exists()
    assert not optimization_info_path.exists()

    motif_summary = pandas.read_csv(motif_summary_path)
    assert motif_summary.to_dict("list") == {
        "allele": ["HLA-A*02:01"],
        "value": [1.0],
    }
    percent_ranks = pandas.read_csv(tmp_path / "percent_ranks.csv")
    assert "HLA-A*02:01" in percent_ranks.columns


def test_load_accepts_legacy_class1_pseudosequences_file(tmp_path):
    (tmp_path / "manifest.csv").write_text("model_name,allele,config_json\n")
    (tmp_path / LEGACY_CLASS1_PSEUDOSEQUENCES_FILENAME).write_text(
        "allele,pseudosequence\n"
        "HLA-A*02:01,YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY\n"
    )

    predictor = Class1AffinityPredictor.load(
        str(tmp_path),
        optimization_level=0,
    )

    assert predictor.allele_to_sequence == {
        "HLA-A*02:01": "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY",
    }


@pytest.mark.parametrize("filename,sequence", [
    (
        PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[34],
        "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY",
    ),
    (
        PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[37],
        "YFAMYGEKVAHTHVDTLYGVRYDHYYTWAVLAYTWYA",
    ),
    (
        PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[39],
        "YFGERAMPYGEKVAHTHVDTLYGVRYHYYTWAVLAYTWY",
    ),
])
def test_load_accepts_named_pseudosequence_files(tmp_path, filename, sequence):
    (tmp_path / "manifest.csv").write_text("model_name,allele,config_json\n")
    (tmp_path / filename).write_text(
        "allele,pseudosequence\n"
        "HLA-A*02:01,%s\n" % sequence
    )

    predictor = Class1AffinityPredictor.load(
        str(tmp_path),
        optimization_level=0,
    )

    assert predictor.allele_to_sequence == {"HLA-A*02:01": sequence}


@pytest.mark.parametrize("sequence,expected_filename", [
    (
        "YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY",
        PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[34],
    ),
    (
        "YFAMYGEKVAHTHVDTLYGVRYDHYYTWAVLAYTWYA",
        PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[37],
    ),
    (
        "YFGERAMPYGEKVAHTHVDTLYGVRYHYYTWAVLAYTWY",
        PSEUDOSEQUENCE_FILENAMES_BY_LENGTH[39],
    ),
])
def test_save_writes_named_pseudosequence_alias(
        tmp_path, sequence, expected_filename):
    predictor = Class1AffinityPredictor(
        allele_to_sequence={"HLA-A*02:01": sequence},
    )

    predictor.save(str(tmp_path))

    legacy_df = pandas.read_csv(
        tmp_path / LEGACY_ALLELE_SEQUENCES_FILENAME)
    assert legacy_df.to_dict("list") == {
        "allele": ["HLA-A*02:01"],
        "sequence": [sequence],
    }
    named_df = pandas.read_csv(tmp_path / expected_filename)
    assert named_df.to_dict("list") == {
        "allele": ["HLA-A*02:01"],
        "pseudosequence": [sequence],
    }


def test_percent_rank_calibrated_allele_direct_equivalent_missing():
    transform = PercentRankTransform()
    transform.fit(numpy.array([10.0, 20.0, 30.0]), bins=3)
    predictor = Class1AffinityPredictor(
        allele_to_sequence={
            "HLA-A*02:01": "SEQUENCE1",
            "HLA-A*02:02": "SEQUENCE1",
            "HLA-B*07:02": "SEQUENCE2",
        },
        allele_to_percent_rank_transform={"HLA-A*02:01": transform},
    )

    assert (
        predictor.percent_rank_calibrated_allele("HLA-A*02:01")
        == "HLA-A*02:01"
    )
    assert (
        predictor.percent_rank_calibrated_allele("HLA-A*02:02")
        == "HLA-A*02:01"
    )
    assert predictor.percent_rank_calibrated_allele("HLA-B*07:02") is None
    assert predictor.percent_rank_calibrated_allele("HLA-C*03:04") is None


def test_missing_percent_rank_error_reports_calibration_command(tmp_path):
    models_dir = str(tmp_path / "models with spaces")
    predictor = Class1AffinityPredictor(
        allele_to_sequence={"HLA-A*02:01": "SEQUENCE1"},
        provenance_string="generated on test date",
        models_dir=models_dir,
    )

    with pytest.raises(ValueError) as err:
        predictor.percentile_ranks([50.0], allele="HLA-A*02:01")

    message = str(err.value)
    assert "Missing percentile-rank calibration for HLA-A*02:01" in message
    assert (
        "Affinity predictions are available; percentile ranks are not."
        in message
    )
    assert "generated on test date" not in message
    assert "0 model(s)" not in message
    assert "0 percent-rank calibration(s)" not in message
    assert "Calibrate with:" in message
    assert (
        "mhcflurry-calibrate-percentile-ranks --models-dir %s "
        "--allele %s ..." % (
            shlex.quote(models_dir),
            shlex.quote("HLA-A*02:01"),
        )
    ) in message


def test_missing_percent_rank_error_infers_models_dir_from_loaded_models(tmp_path):
    models_dir = str(tmp_path / "loaded models")
    network = Class1NeuralNetwork.from_config(
        {"hyperparameters": {}},
        weight_paths=str(tmp_path / "loaded models" / "weights_PAN-CLASS1-0.npz"),
    )
    assert "network_weight_paths" not in network.get_config()
    predictor = Class1AffinityPredictor(
        class1_pan_allele_models=[network],
        allele_to_sequence={"HLA-A*02:01": "SEQUENCE1"},
    )

    assert predictor.models_dir is None
    assert predictor.models_dir_for_diagnostics() == models_dir

    with pytest.raises(ValueError) as err:
        predictor.percentile_ranks([50.0], allele="HLA-A*02:01")

    message = str(err.value)
    assert "Calibrate with:" in message
    assert (
        "mhcflurry-calibrate-percentile-ranks --models-dir %s "
        "--allele %s ..." % (
            shlex.quote(models_dir),
            shlex.quote("HLA-A*02:01"),
        )
    ) in message


def predict_and_check(
    allele, peptide, predictor=DOWNLOADED_PREDICTOR, expected_range=(0, 500)
):
    def debug():
        print(
            "\n%s"
            % (
                predictor.predict_to_dataframe(
                    peptides=[peptide],
                    allele=allele,
                    include_individual_model_predictions=True,
                )
            )
        )

        (prediction,) = predictor.predict(allele=allele, peptides=[peptide])
        assert prediction >= expected_range[0], (predictor, prediction, debug())
        assert prediction <= expected_range[1], (predictor, prediction, debug())


@pytest.mark.slow
@pytest.mark.integration
def test_a1_known_epitopes_in_newly_trained_model():
    allele = "HLA-A*01:01"
    df = pandas.read_csv(
        get_path("data_curated", "curated_training_data.affinity.csv.bz2")
    )
    df = df.loc[
        (df.allele == allele)
        & (df.peptide.str.len() >= 8)
        & (df.peptide.str.len() <= 15)
    ]

    hyperparameters = {
        "max_epochs": 100,
        "patience": 10,
        "early_stopping": True,
        "validation_split": 0.2,
        "random_negative_rate": 0.0,
        "random_negative_constant": 25,
        "peptide_amino_acid_encoding": "BLOSUM62",
        "use_embedding": False,
        "kmer_size": 15,
        "batch_normalization": False,
        "locally_connected_layers": [
            {"filters": 8, "activation": "tanh", "kernel_size": 3}
        ],
        "activation": "relu",
        "output_activation": "sigmoid",
        "layer_sizes": [32],
        "random_negative_affinity_min": 20000.0,
        "random_negative_affinity_max": 50000.0,
        "dense_layer_l1_regularization": 0.001,
        "dropout_probability": 0.0,
    }

    predictor = Class1AffinityPredictor()
    predictor.fit_allele_specific_predictors(
        n_models=2,
        architecture_hyperparameters_list=[hyperparameters],
        allele=allele,
        peptides=df.peptide.values,
        affinities=df.measurement_value.values,
        verbose=0,
    )

    predict_and_check("HLA-A*01:01", "EVDPIGHLY", predictor=predictor)

    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    predictor.save(models_dir)
    predictor2 = Class1AffinityPredictor.load(models_dir)
    predict_and_check("HLA-A*01:01", "EVDPIGHLY", predictor=predictor2)
    shutil.rmtree(models_dir)

    predictor3 = Class1AffinityPredictor(
        allele_to_allele_specific_models={
            allele: [predictor.allele_to_allele_specific_models[allele][0]]
        }
    )
    predict_and_check("HLA-A*01:01", "EVDPIGHLY", predictor=predictor3)
    models_dir = tempfile.mkdtemp("_models")
    print(models_dir)
    predictor3.save(models_dir)
    predictor4 = Class1AffinityPredictor.load(models_dir)
    predict_and_check("HLA-A*01:01", "EVDPIGHLY", predictor=predictor4)
    shutil.rmtree(models_dir)


@pytest.mark.slow
@pytest.mark.integration
def test_class1_affinity_predictor_a0205_memorize_training_data():
    # Memorize the dataset.
    hyperparameters = dict(
        activation="tanh",
        layer_sizes=[64],
        max_epochs=100,
        early_stopping=False,
        validation_split=0.0,
        locally_connected_layers=[],
        dense_layer_l1_regularization=0.0,
        dropout_probability=0.0,
    )

    allele = "HLA-A*02:05"

    df = pandas.read_csv(
        get_path("data_curated", "curated_training_data.affinity.csv.bz2")
    )
    df = df.loc[df.allele == allele]
    df = df.loc[df.peptide.str.len() == 9]
    df = df.loc[df.measurement_type == "quantitative"]
    df = df.loc[df.measurement_source == "kim2014"]

    predictor = Class1AffinityPredictor()
    predictor.fit_allele_specific_predictors(
        n_models=2,
        architecture_hyperparameters_list=[hyperparameters],
        allele=allele,
        peptides=df.peptide.values,
        affinities=df.measurement_value.values,
        verbose=0,
    )
    predictor.calibrate_percentile_ranks(num_peptides_per_length=1000)
    ic50_pred = predictor.predict(df.peptide.values, allele=allele)
    ic50_true = df.measurement_value.values
    assert len(ic50_pred) == len(ic50_true)
    testing.assert_allclose(
        numpy.log(ic50_pred), numpy.log(ic50_true), rtol=0.2, atol=0.2
    )

    ic50_pred_df = predictor.predict_to_dataframe(df.peptide.values, allele=allele)
    print(ic50_pred_df)
    assert "prediction_percentile" in ic50_pred_df.columns
    assert ic50_pred_df.prediction_percentile.isnull().sum() == 0

    ic50_pred_df2 = predictor.predict_to_dataframe(
        df.peptide.values, allele=allele, include_individual_model_predictions=True
    )
    print(ic50_pred_df2)

    # Test an unknown allele
    print("Starting unknown allele check")
    assert predictor.supported_alleles == [allele]
    ic50_pred = predictor.predict(df.peptide.values, allele="HLA-A*02:01", throw=False)
    assert numpy.isnan(ic50_pred).all()

    testing.assert_raises(
        ValueError, predictor.predict, df.peptide.values, allele="HLA-A*02:01"
    )

    assert predictor.supported_alleles == [allele]
    testing.assert_raises(ValueError, predictor.predict, ["AAAAA"], allele=allele)  # too short
    testing.assert_raises(
        ValueError,
        predictor.predict,
        ["AAAAAAAAAAAAAAAAAAAA"],  # too long
        allele=allele,
    )
    ic50_pred = predictor.predict(
        ["AAAAA", "AAAAAAAAA", "AAAAAAAAAAAAAAAAAAAA"], allele=allele, throw=False
    )
    assert numpy.isnan(ic50_pred[0])
    assert not numpy.isnan(ic50_pred[1])
    assert numpy.isnan(ic50_pred[2])


def test_no_nans():
    df = DOWNLOADED_PREDICTOR.predict_to_dataframe(
        alleles=["A02:01", "A02:02"], peptides=["SIINFEKL", "SIINFEKLL"]
    )
    print(df)
    assert not df.isnull().any().any()


def test_predict_implementations_equivalent():
    for allele in ["HLA-A02:01", "A02:02"]:
        for centrality_measure in ["mean", "robust_mean"]:
            peptides = ["SIINFEKL", "SYYNFIIIKL", "SIINKFELQY"]

            pred1 = DOWNLOADED_PREDICTOR.predict(
                allele=allele,
                peptides=peptides + ["SSSN"],
                throw=False,
                centrality_measure=centrality_measure,
            )
            pred2 = DOWNLOADED_PREDICTOR.predict_to_dataframe(
                allele=allele,
                peptides=peptides + ["SSSN"],
                throw=False,
                centrality_measure=centrality_measure,
            ).prediction.values
            testing.assert_almost_equal(pred1, pred2, decimal=2)

            pred1 = DOWNLOADED_PREDICTOR.predict(
                allele=allele, peptides=peptides, centrality_measure=centrality_measure
            )
            pred2 = DOWNLOADED_PREDICTOR.predict_to_dataframe(
                allele=allele, peptides=peptides, centrality_measure=centrality_measure
            ).prediction.values
            testing.assert_almost_equal(pred1, pred2, decimal=2)


def test_no_runtime_warnings_for_unsupported_rows():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        df = DOWNLOADED_PREDICTOR.predict_to_dataframe(
            allele="HLA-A*02:01",
            peptides=["SIINFEKL", "SSSN"],
            throw=False,
            include_confidence_intervals=True,
            centrality_measure="mean",
        )
        df2 = DOWNLOADED_PREDICTOR.predict_to_dataframe(
            allele="HLA-A*02:01",
            peptides=["SSSN"],
            throw=False,
            include_confidence_intervals=True,
            centrality_measure="robust_mean",
        )
    assert not numpy.isnan(df.loc[df.peptide == "SIINFEKL", "prediction"].iloc[0])
    assert numpy.isnan(df.loc[df.peptide == "SSSN", "prediction"].iloc[0])
    assert numpy.isnan(df2["prediction"].iloc[0])
