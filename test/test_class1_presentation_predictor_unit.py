import numpy
import pandas
import torch

import mhcflurry.class1_presentation_predictor as presentation_module
import mhcflurry.train_presentation_models_command as train_presentation
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


class FakeCartesianAffinityPredictor(FakeAffinityPredictor):
    def __init__(self, scores):
        super().__init__(scores)
        self.cartesian_call_shapes = []

    def predict_cartesian_pan_allele(
            self,
            peptides,
            alleles,
            throw=True,
            model_kwargs=None):
        del throw, model_kwargs
        seqs = list(peptides.sequences)
        alleles = list(alleles)
        self.cartesian_call_shapes.append((len(seqs), len(alleles)))
        return numpy.asarray([
            [self.scores[(allele, peptide)] for allele in alleles]
            for peptide in seqs
        ], dtype=float)


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


def test_predict_affinity_uses_cartesian_fast_path(monkeypatch):
    monkeypatch.setattr(
        presentation_module, "_PRESENTATION_PREDICT_TARGET_ROWS", 4,
    )
    scores = {
        ("A", "P0"): 3.0,
        ("A", "P1"): 10.0,
        ("B", "P0"): 5.0,
        ("B", "P1"): 1.0,
        ("C", "P0"): 2.0,
        ("C", "P1"): 9.0,
    }
    affinity_predictor = FakeCartesianAffinityPredictor(scores)
    predictor = Class1PresentationPredictor(
        affinity_predictor=affinity_predictor,
    )

    result = predictor.predict_affinity(
        peptides=["P0", "P1"],
        alleles={
            "s1": ["B", "A"],
            "s2": ["C", "A"],
        },
        include_affinity_percentile=False,
        verbose=0,
    )

    assert result.affinity.tolist() == [3.0, 1.0, 2.0, 9.0]
    assert result.best_allele.tolist() == ["A", "B", "C", "C"]
    assert affinity_predictor.cartesian_call_shapes
    assert affinity_predictor.call_sizes == []


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


def test_fit_from_scores_trains_expected_variants():
    predictor = Class1PresentationPredictor()

    predictor.fit_from_scores(
        targets=[0, 1, 0, 1, 0, 1],
        affinities=[50000, 50, 30000, 100, 40000, 75],
        processing_scores_by_model={
            "without_flanks": [0.1, 0.9, 0.2, 0.7, 0.1, 0.8],
            "with_flanks": [0.2, 0.8, 0.3, 0.6, 0.2, 0.7],
        },
        verbose=0,
    )

    assert set(predictor.weights_dataframe.index) == {
        "without_flanks",
        "with_flanks",
    }
    assert set(predictor.weights_dataframe.columns) == {
        "intercept",
        "affinity_score",
        "processing_score",
    }


class FakePresentationPredictor:
    def predict_affinity(
            self,
            peptides,
            alleles,
            sample_names=None,
            include_affinity_percentile=False,
            verbose=0,
            model_kwargs=None):
        del alleles, sample_names, include_affinity_percentile, verbose
        assert model_kwargs == {"batch_size": "auto"}
        return pandas.DataFrame({
            "affinity": numpy.arange(len(peptides), dtype=float) + 10.0,
        })

    def predict_processing(
            self,
            peptides,
            n_flanks=None,
            c_flanks=None,
            verbose=0,
            batch_size="auto"):
        del c_flanks, verbose
        assert batch_size == "auto"
        offset = 100.0 if n_flanks is None else 200.0
        return numpy.arange(len(peptides), dtype=float) + offset


def test_presentation_feature_chunks_use_global_data():
    df = pandas.DataFrame({
        "experiment_id": ["s1", "s1", "s1", "s2", "s2"],
        "peptide": ["A", "B", "C", "D", "E"],
        "n_flank": ["N"] * 5,
        "c_flank": ["C"] * 5,
    })
    work_items = train_presentation.make_feature_work_items(df, chunk_size=2)
    assert [
        (item["start"], item["end"], item["sample_name"])
        for item in work_items
    ] == [
        (0, 2, "s1"),
        (2, 3, "s1"),
        (3, 5, "s2"),
    ]

    train_presentation.GLOBAL_DATA.clear()
    train_presentation.GLOBAL_DATA.update({
        "predictor": FakePresentationPredictor(),
        "data": df,
        "experiment_to_alleles": {
            "s1": ["A*01:01"],
            "s2": ["A*02:01"],
        },
    })
    result = train_presentation.predict_feature_chunk(
        chunk_num=0,
        start=0,
        end=2,
        sample_name="s1",
        include_without_flanks=True,
        include_with_flanks=True,
    )

    numpy.testing.assert_equal(result["affinity"], [10.0, 11.0])
    numpy.testing.assert_equal(
        result["processing_scores"]["without_flanks"], [100.0, 101.0])
    numpy.testing.assert_equal(
        result["processing_scores"]["with_flanks"], [200.0, 201.0])


class FakeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4))


class FakeNetworkModel:
    def __init__(self, network):
        self._network = network

    def network(self, borrow=False):
        del borrow
        return self._network


def test_estimate_presentation_feature_worker_gb_uses_model_shape(monkeypatch):
    class FakeAffinity:
        allele_to_allele_specific_models = {}

        def __init__(self, network):
            self.class1_pan_allele_models = [FakeNetworkModel(network)]

    class FakeProcessingPredictor:
        def __init__(self, network):
            self.models = [FakeNetworkModel(network)]

    class FakePresentation:
        def __init__(self, affinity_network, processing_network):
            self.affinity_predictor = FakeAffinity(affinity_network)
            self.processing_predictor_without_flanks = FakeProcessingPredictor(
                processing_network)
            self.processing_predictor_with_flanks = None

    affinity_network = FakeNetwork()
    processing_network = FakeNetwork()
    monkeypatch.setenv("MHCFLURRY_PRESENTATION_WORKER_RUNTIME_FLOOR_GB", "1")
    monkeypatch.setenv("MHCFLURRY_PRESENTATION_WORKER_SAFETY_FACTOR", "1")
    monkeypatch.setenv("MHCFLURRY_PRESENTATION_WORKER_TRANSIENT_ROWS", "10")
    monkeypatch.setattr(
        train_presentation,
        "_estimate_peak_bytes_per_row",
        lambda network: 512 * 1024 * 1024
        if network is processing_network else 1024,
    )

    estimate = train_presentation.estimate_presentation_feature_worker_gb(
        type("Args", (), {"feature_chunk_size": 20})(),
        FakePresentation(affinity_network, processing_network),
    )

    assert 5.9 < estimate < 6.1
