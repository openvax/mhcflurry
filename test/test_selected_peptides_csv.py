"""
Regression checks against selected-peptides.csv.

Compares current MHCflurry predictions to values recorded from the
previous public release, and checks NetMHCpan affinity is reasonably close.
"""
import os

import numpy as np
import pandas as pd
import pytest

from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
from mhcflurry.downloads import get_path
from mhcflurry.testing_utils import startup, cleanup


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "selected-peptides.csv",
)


def _normalize_allele(allele):
    if allele is None or (isinstance(allele, float) and np.isnan(allele)):
        return allele
    allele = str(allele).strip()
    if allele.startswith("HLA-") or "-" in allele:
        return allele
    if "*" in allele:
        return f"HLA-{allele}"
    return allele


@pytest.fixture(scope="module")
def selected_peptides_predictions():
    startup()
    try:
        df = pd.read_csv(DATA_PATH)
        peptides = df["mhcflurry_peptide"].fillna(df["peptide"]).tolist()
        alleles = [_normalize_allele(a) for a in df["mhcflurry_best_allele"].tolist()]

        sample_names = [f"row_{i}" for i in range(len(peptides))]
        alleles_dict = {name: [allele] for name, allele in zip(sample_names, alleles)}

        predictor = Class1PresentationPredictor.load(
            get_path("models_class1_presentation", "models")
        )
        pred_df = predictor.predict(
            peptides=peptides,
            alleles=alleles_dict,
            sample_names=sample_names,
            n_flanks=None,
            c_flanks=None,
            verbose=0,
        )
        pred_df = pred_df.sort_values("peptide_num").reset_index(drop=True)
        df = df.reset_index(drop=True)
        return df, pred_df
    finally:
        cleanup()


def test_selected_peptides_mhcflurry_matches_csv(selected_peptides_predictions):
    df, pred_df = selected_peptides_predictions

    np.testing.assert_allclose(
        pred_df["affinity"].values,
        df["mhcflurry_affinity"].values.astype(float),
        rtol=0.01,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pred_df["processing_score"].values,
        df["mhcflurry_processing_score"].values.astype(float),
        rtol=0.01,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pred_df["presentation_score"].values,
        df["mhcflurry_presentation_score"].values.astype(float),
        rtol=0.01,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pred_df["presentation_percentile"].values,
        df["mhcflurry_presentation_percentile"].values.astype(float),
        rtol=0.01,
        atol=0.1,
    )


def test_selected_peptides_netmhcpan_affinity_close(selected_peptides_predictions):
    df, _ = selected_peptides_predictions
    net_alleles = [_normalize_allele(a) for a in df["netmhcpan_best_allele_by_pr"].tolist()]
    mhc_alleles = [_normalize_allele(a) for a in df["mhcflurry_best_allele"].tolist()]

    mask = [
        (net.startswith("HLA-A") or net.startswith("HLA-B"))
        and (mhc.startswith("HLA-A") or mhc.startswith("HLA-B"))
        for net, mhc in zip(net_alleles, mhc_alleles)
    ]

    df = df.loc[mask].reset_index(drop=True)
    peptides = df["mhcflurry_peptide"].fillna(df["peptide"]).tolist()
    alleles = [net for net, keep in zip(net_alleles, mask) if keep]

    startup()
    try:
        predictor = Class1AffinityPredictor.load(
            get_path("models_class1", "models")
        )
        mhc_aff = predictor.predict(peptides=peptides, alleles=alleles).astype(float)
    finally:
        cleanup()

    net_aff = df["netmhcpan_aff"].values.astype(float)
    mhc_aff = np.clip(mhc_aff, 1e-6, None)
    net_aff = np.clip(net_aff, 1e-6, None)
    log_diff = np.abs(np.log10(mhc_aff) - np.log10(net_aff))

    # Within 10x for HLA-A/B when comparing to NetMHCpan's best allele.
    assert (log_diff <= np.log10(10)).all()
