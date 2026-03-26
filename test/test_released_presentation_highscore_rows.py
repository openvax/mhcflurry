"""
Regression tests for released presentation model predictions.

Expected values cover peptide+flank contexts where at least one allele
has a high presentation score (>0.9), including low-scoring alleles for
the same contexts.  Values were generated from the TF/Keras implementation
and are stored under test/data/.
"""
import os
import warnings

import numpy as np
import pandas as pd

from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
from mhcflurry.testing_utils import startup, cleanup


warnings.filterwarnings(
    "ignore",
    message=r".*Downcasting behavior in `replace` is deprecated.*",
    category=FutureWarning,
)

EXPECTED_CSV = "master_released_class1_presentation_highscore_rows.csv.gz"
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


def _load_expected():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    return pd.read_csv(
        os.path.join(data_dir, EXPECTED_CSV), keep_default_na=False)


def _atol_for_output(column):
    if "percentile" in column:
        return 1e-5
    if "affinity" in column:
        return 0.1
    return 1e-5


def test_expected_data_has_high_and_low_contexts():
    """Sanity-check that the expected data spans a wide score range."""
    expected_df = _load_expected()
    by_context = expected_df.groupby(
        ["peptide", "n_flank", "c_flank"], observed=True)
    context_max = by_context[HIGH_SCORE_COLUMNS].max()
    context_min = by_context[HIGH_SCORE_COLUMNS].min()

    assert (
        (context_max["pres_with_presentation_score"] > 0.9)
        | (context_max["pres_without_presentation_score"] > 0.9)
    ).all()
    assert (context_min["pres_with_presentation_score"] < 0.2).all()
    assert (context_min["pres_without_presentation_score"] < 0.2).all()


def test_presentation_predictions():
    expected_df = _load_expected()

    affinity_predictor = Class1AffinityPredictor.load()
    presentation_predictor = Class1PresentationPredictor.load()

    peptides = expected_df["peptide"].tolist()
    alleles = expected_df["allele"].tolist()
    n_flanks = expected_df["n_flank"].tolist()
    c_flanks = expected_df["c_flank"].tolist()

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
    np.testing.assert_array_equal(
        aff_df["peptide"].to_numpy(), expected_df["peptide"].to_numpy())
    np.testing.assert_array_equal(
        aff_df["allele"].to_numpy(), expected_df["allele"].to_numpy())

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

    predicted = expected_df[BASE_COLUMNS].copy()
    predicted["affinity_prediction"] = aff_df["prediction"].values
    predicted["affinity_prediction_low"] = aff_df.get("prediction_low", np.nan)
    predicted["affinity_prediction_high"] = aff_df.get("prediction_high", np.nan)
    predicted["affinity_prediction_percentile"] = aff_df.get(
        "prediction_percentile", np.nan)

    predicted["pres_with_affinity"] = pres_with_df["affinity"].values
    predicted["pres_with_best_allele"] = (
        pres_with_df["best_allele"].astype(str).values)
    predicted["pres_with_affinity_percentile"] = (
        pres_with_df["affinity_percentile"].values)
    predicted["processing_with_score"] = pres_with_df["processing_score"].values
    predicted["pres_with_processing_score"] = (
        pres_with_df["processing_score"].values)
    predicted["pres_with_presentation_score"] = (
        pres_with_df["presentation_score"].values)
    predicted["pres_with_presentation_percentile"] = (
        pres_with_df["presentation_percentile"].values)

    predicted["pres_without_affinity"] = pres_without_df["affinity"].values
    predicted["pres_without_best_allele"] = (
        pres_without_df["best_allele"].astype(str).values)
    predicted["pres_without_affinity_percentile"] = (
        pres_without_df["affinity_percentile"].values)
    predicted["processing_without_score"] = (
        pres_without_df["processing_score"].values)
    predicted["pres_without_processing_score"] = (
        pres_without_df["processing_score"].values)
    predicted["pres_without_presentation_score"] = (
        pres_without_df["presentation_score"].values)
    predicted["pres_without_presentation_percentile"] = (
        pres_without_df["presentation_percentile"].values)

    for col in STRING_COLUMNS:
        np.testing.assert_array_equal(
            predicted[col].astype(str).to_numpy(),
            expected_df[col].astype(str).to_numpy(),
        )

    numeric_columns = [
        c for c in expected_df.columns
        if c not in BASE_COLUMNS + STRING_COLUMNS
    ]
    for col in numeric_columns:
        np.testing.assert_allclose(
            predicted[col].to_numpy(dtype=np.float64),
            expected_df[col].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=_atol_for_output(col),
        )
