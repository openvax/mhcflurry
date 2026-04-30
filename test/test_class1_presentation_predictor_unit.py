import numpy
import pandas

import mhcflurry.class1_presentation_predictor as presentation_module
from mhcflurry.class1_presentation_predictor import Class1PresentationPredictor


class FakeAffinityPredictor:
    def __init__(self, scores):
        self.scores = scores
        self.call_sizes = []

    def predict(self, peptides, alleles, model_kwargs=None, throw=True):
        seqs = list(peptides.sequences)
        alleles = list(alleles)
        self.call_sizes.append(len(seqs))
        return numpy.asarray([
            self.scores[(allele, peptide)]
            for allele, peptide in zip(alleles, seqs)
        ], dtype=float)

    def percentile_ranks(self, affinities, alleles=None, throw=False):
        return numpy.zeros(len(affinities), dtype=float)


def test_predict_affinity_sample_names_none_streams_best_by_sample(monkeypatch):
    monkeypatch.setattr(
        presentation_module, "_PRESENTATION_PREDICT_TARGET_ROWS", 4,
    )
    scores = {
        ("A", "P0"): 5.0,
        ("A", "P1"): 10.0,
        ("A", "P2"): numpy.nan,
        ("B", "P0"): 5.0,
        ("B", "P1"): 1.0,
        ("B", "P2"): numpy.nan,
        ("C", "P0"): 2.0,
        ("C", "P1"): 9.0,
        ("C", "P2"): numpy.nan,
    }
    affinity_predictor = FakeAffinityPredictor(scores)
    predictor = Class1PresentationPredictor(
        affinity_predictor=affinity_predictor,
    )

    result = predictor.predict_affinity(
        peptides=["P0", "P1", "P2"],
        alleles={
            "s1": ["B", "A"],
            "s2": ["C", "A"],
        },
        include_affinity_percentile=False,
        verbose=0,
    )

    expected = pandas.DataFrame({
        "peptide": ["P0", "P1", "P2", "P0", "P1", "P2"],
        "peptide_num": [0, 1, 2, 0, 1, 2],
        "sample_name": ["s1", "s1", "s1", "s2", "s2", "s2"],
        "affinity": [5.0, 1.0, numpy.nan, 2.0, 9.0, numpy.nan],
        "best_allele": ["B", "B", None, "C", "C", None],
    })
    pandas.testing.assert_frame_equal(result, expected)
    assert affinity_predictor.call_sizes
    assert max(affinity_predictor.call_sizes) <= 4


def test_predict_affinity_sample_names_none_handles_no_peptides():
    affinity_predictor = FakeAffinityPredictor({})
    predictor = Class1PresentationPredictor(
        affinity_predictor=affinity_predictor,
    )

    result = predictor.predict_affinity(
        peptides=[],
        alleles={"s1": ["A"], "s2": ["B"]},
        include_affinity_percentile=False,
        verbose=0,
    )

    assert list(result.columns) == [
        "peptide", "peptide_num", "sample_name", "affinity", "best_allele",
    ]
    assert result.empty
    assert affinity_predictor.call_sizes == []
