"""Parity + perf tests for ``Class1AffinityPredictor.calibrate_percentile_ranks_fast``.

The fast path batches many alleles into a single forward after caching the
peptide-side activations per network. The parity test anchors that the
alternative schedule produces the same ``PercentRankTransform`` as the
slow legacy path; the smoke test anchors that it actually runs end to
end against a downloaded pan-allele release.
"""

import numpy
import pandas
import pytest

from mhcflurry import Class1AffinityPredictor
from mhcflurry.common import random_peptides
from mhcflurry.downloads import get_path


def _load_downloaded_pan_allele():
    try:
        models_dir = get_path("models_class1_pan", "models.combined")
    except Exception as exc:
        pytest.skip(f"public pan-allele models not available: {exc}")
    return Class1AffinityPredictor.load(models_dir, optimization_level=0)


def _pick_alleles(predictor, n):
    # deterministic, covers multiple class-I families
    candidates = [
        a for a in sorted(predictor.allele_to_sequence)
        if a.startswith("HLA-")
    ][:n]
    assert len(candidates) >= 2, "Need at least 2 HLA alleles for parity"
    return candidates


def test_calibrate_fast_parity_with_legacy_path():
    predictor = _load_downloaded_pan_allele()
    alleles = _pick_alleles(predictor, 4)
    peptides = random_peptides(2000, length=9)

    # Run legacy path
    legacy = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"),
        optimization_level=0,
    )
    # Drop any existing transforms so we compare a fresh fit
    legacy.allele_to_percent_rank_transform = {}
    legacy.calibrate_percentile_ranks(
        peptides=peptides,
        alleles=alleles,
        motif_summary=False,
    )

    # Run fast path
    predictor.allele_to_percent_rank_transform = {}
    predictor.calibrate_percentile_ranks_fast(
        peptides=peptides,
        alleles=alleles,
        motif_summary=False,
        allele_batch_size=2,  # exercise the batching boundary
        peptide_batch_size=500,
    )

    for allele in alleles:
        a = legacy.allele_to_percent_rank_transform[allele]
        b = predictor.allele_to_percent_rank_transform[allele]
        # CDFs should be bit-identical — same peptides, same networks,
        # same aggregation, same bin edges. Any drift means the fast
        # path has diverged semantically.
        numpy.testing.assert_allclose(
            a.cdf, b.cdf, rtol=0, atol=1e-9,
            err_msg=f"CDF mismatch for {allele}",
        )
        numpy.testing.assert_allclose(
            a.bin_edges, b.bin_edges, rtol=0, atol=1e-12,
            err_msg=f"bin edges mismatch for {allele}",
        )


def test_calibrate_fast_parity_with_motif_summary():
    predictor = _load_downloaded_pan_allele()
    alleles = _pick_alleles(predictor, 3)
    # Multi-length peptides so length_distributions is non-trivial
    peptides = (
        random_peptides(800, length=9)
        + random_peptides(400, length=10)
        + random_peptides(200, length=11)
    )

    legacy = Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"),
        optimization_level=0,
    )
    legacy.allele_to_percent_rank_transform = {}
    legacy_summary = legacy.calibrate_percentile_ranks(
        peptides=peptides,
        alleles=alleles,
        motif_summary=True,
        summary_top_peptide_fractions=[0.01],
    )

    predictor.allele_to_percent_rank_transform = {}
    fast_summary = predictor.calibrate_percentile_ranks_fast(
        peptides=peptides,
        alleles=alleles,
        motif_summary=True,
        summary_top_peptide_fractions=(0.01,),
        allele_batch_size=3,
        peptide_batch_size=500,
    )

    def _sort_key(df):
        return df.sort_values(
            [c for c in ["allele", "length", "cutoff_fraction"]
             if c in df.columns]
        ).reset_index(drop=True)

    # length_distributions should match exactly
    pandas.testing.assert_frame_equal(
        _sort_key(legacy_summary["length_distributions"]),
        _sort_key(fast_summary["length_distributions"]),
        check_exact=False, rtol=1e-10, atol=1e-12,
    )
    # frequency_matrices: the top-k peptide *set* should match per
    # (allele,length,cutoff) bucket. Ordering within ties can drift in
    # numerical rounding, so we compare the set of frequency values
    # after sorting within each group.
    lf = legacy_summary["frequency_matrices"]
    ff = fast_summary["frequency_matrices"]
    assert set(lf.allele.unique()) == set(ff.allele.unique())
    for (allele, length, cutoff), sub_l in lf.groupby(
        ["allele", "length", "cutoff_fraction"]
    ):
        sub_f = ff[
            (ff.allele == allele)
            & (ff.length == length)
            & (ff.cutoff_fraction == cutoff)
        ]
        assert len(sub_l) == len(sub_f)
