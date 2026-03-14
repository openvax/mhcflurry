"""
Compare released presentation predictions against high-score TF fixtures.

The fixture keeps all alleles for peptide+flank contexts where at least one
allele has a high TF presentation score (>0.9), so low-scoring alleles for the
same contexts are also tested.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
from mhcflurry.downloads import (
    configure,
    get_current_release,
    get_default_class1_models_dir,
    get_default_class1_presentation_models_dir,
)
from mhcflurry.testing_utils import startup, cleanup


warnings.filterwarnings(
    "ignore",
    message=r".*Downcasting behavior in `replace` is deprecated.*",
    category=FutureWarning,
)


FIXTURE_CSV = "master_released_class1_presentation_highscore_rows.csv.gz"
FIXTURE_METADATA = "master_released_class1_presentation_highscore_rows_metadata.json"
BASE_COLUMNS = ["row_id", "peptide", "allele", "n_flank", "c_flank"]
STRING_COLUMNS = ["pres_with_best_allele", "pres_without_best_allele"]
HIGH_SCORE_COLUMNS = [
    "pres_with_presentation_score",
    "pres_without_presentation_score",
]


def setup_module():
    startup()


def teardown_module():
    cleanup()


def _load_fixture():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    df = pd.read_csv(os.path.join(data_dir, FIXTURE_CSV), keep_default_na=False)
    with open(os.path.join(data_dir, FIXTURE_METADATA), "r") as inp:
        metadata = json.load(inp)
    return df, metadata


def _skip_if_fixture_incompatible(metadata, presentation_predictor):
    fixture_release = metadata.get("release")
    current_release = get_current_release()
    if fixture_release and fixture_release != current_release:
        pytest.skip(
            "Fixture was generated for release %s, current downloads are %s."
            % (fixture_release, current_release)
        )

    fixture_prov = metadata.get("presentation_provenance")
    if fixture_prov and fixture_prov != presentation_predictor.provenance_string:
        pytest.skip(
            "Fixture presentation provenance is %s, current predictor is %s."
            % (fixture_prov, presentation_predictor.provenance_string)
        )

    fixture_aff_prov = metadata.get("presentation_internal_affinity_provenance")
    current_aff_prov = presentation_predictor.affinity_predictor.provenance_string
    if fixture_aff_prov and fixture_aff_prov != current_aff_prov:
        pytest.skip(
            "Fixture internal affinity provenance is %s, current predictor is %s."
            % (fixture_aff_prov, current_aff_prov)
        )


def _on_gpu():
    import torch
    from mhcflurry.common import get_pytorch_device
    return get_pytorch_device().type != "cpu"


def _atol_for_output(column):
    # GPU float32 arithmetic can differ from CPU by up to ~1e-4 for scores.
    gpu = _on_gpu()
    if "percentile" in column:
        return 1e-3 if gpu else 1e-6
    if "affinity" in column:
        return 0.1 if gpu else 0.05
    return 1e-3 if gpu else 1e-6


def test_presentation_highscore_fixture_has_high_and_low_contexts():
    fixture_df, _ = _load_fixture()
    context_group = fixture_df.groupby(["peptide", "n_flank", "c_flank"], observed=True)
    context_max = context_group[HIGH_SCORE_COLUMNS].max()
    context_min = context_group[HIGH_SCORE_COLUMNS].min()

    assert (
        (context_max["pres_with_presentation_score"] > 0.9)
        | (context_max["pres_without_presentation_score"] > 0.9)
    ).all()
    assert (context_min["pres_with_presentation_score"] < 0.2).all()
    assert (context_min["pres_without_presentation_score"] < 0.2).all()


def test_released_presentation_predictions_match_highscore_master_fixture():
    fixture_df, metadata = _load_fixture()
    configure()

    affinity_predictor = Class1AffinityPredictor.load(get_default_class1_models_dir())
    presentation_predictor = Class1PresentationPredictor.load(
        get_default_class1_presentation_models_dir()
    )
    _skip_if_fixture_incompatible(metadata, presentation_predictor)

    peptides = fixture_df["peptide"].tolist()
    alleles = fixture_df["allele"].tolist()
    n_flanks = fixture_df["n_flank"].tolist()
    c_flanks = fixture_df["c_flank"].tolist()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Downcasting behavior in `replace` is deprecated.*",
            category=FutureWarning,
        )
        aff_df = affinity_predictor.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            throw=False,
            include_percentile_ranks=True,
            include_confidence_intervals=True,
            centrality_measure="mean",
            model_kwargs={"batch_size": 4096},
        )
    np.testing.assert_array_equal(aff_df["peptide"].to_numpy(), fixture_df["peptide"].to_numpy())
    np.testing.assert_array_equal(aff_df["allele"].to_numpy(), fixture_df["allele"].to_numpy())

    sample_names = alleles
    allele_map = {allele: [allele] for allele in sorted(set(alleles))}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Downcasting behavior in `replace` is deprecated.*",
            category=FutureWarning,
        )
        pres_with_df = presentation_predictor.predict(
            peptides=peptides,
            alleles=allele_map,
            sample_names=sample_names,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            include_affinity_percentile=True,
            verbose=0,
            throw=True,
        ).sort_values("peptide_num")
        pres_without_df = presentation_predictor.predict(
            peptides=peptides,
            alleles=allele_map,
            sample_names=sample_names,
            n_flanks=None,
            c_flanks=None,
            include_affinity_percentile=True,
            verbose=0,
            throw=True,
        ).sort_values("peptide_num")

    predicted = fixture_df[BASE_COLUMNS].copy()
    predicted["affinity_prediction"] = aff_df["prediction"].values
    predicted["affinity_prediction_low"] = aff_df.get("prediction_low", np.nan)
    predicted["affinity_prediction_high"] = aff_df.get("prediction_high", np.nan)
    predicted["affinity_prediction_percentile"] = aff_df.get("prediction_percentile", np.nan)

    predicted["pres_with_affinity"] = pres_with_df["affinity"].values
    predicted["pres_with_best_allele"] = pres_with_df["best_allele"].astype(str).values
    predicted["pres_with_affinity_percentile"] = pres_with_df["affinity_percentile"].values
    predicted["processing_with_score"] = pres_with_df["processing_score"].values
    predicted["pres_with_processing_score"] = pres_with_df["processing_score"].values
    predicted["pres_with_presentation_score"] = pres_with_df["presentation_score"].values
    predicted["pres_with_presentation_percentile"] = pres_with_df[
        "presentation_percentile"
    ].values

    predicted["pres_without_affinity"] = pres_without_df["affinity"].values
    predicted["pres_without_best_allele"] = pres_without_df["best_allele"].astype(str).values
    predicted["pres_without_affinity_percentile"] = pres_without_df[
        "affinity_percentile"
    ].values
    predicted["processing_without_score"] = pres_without_df["processing_score"].values
    predicted["pres_without_processing_score"] = pres_without_df["processing_score"].values
    predicted["pres_without_presentation_score"] = pres_without_df["presentation_score"].values
    predicted["pres_without_presentation_percentile"] = pres_without_df[
        "presentation_percentile"
    ].values

    for col in STRING_COLUMNS:
        np.testing.assert_array_equal(
            predicted[col].astype(str).to_numpy(), fixture_df[col].astype(str).to_numpy()
        )

    numeric_columns = [c for c in fixture_df.columns if c not in BASE_COLUMNS + STRING_COLUMNS]
    for col in numeric_columns:
        np.testing.assert_allclose(
            predicted[col].to_numpy(dtype=np.float64),
            fixture_df[col].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=_atol_for_output(col),
        )
