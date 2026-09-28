"""Numeric/string class-I encoding parity on CPU and available accelerators."""

import numpy as np
import pytest
import torch

from mhcflurry.encodable_sequences import EncodableSequences, EncodingError
from mhcflurry.amino_acid import AMINO_ACID_INDEX
from mhcflurry.numeric_sequences import NumericSequences
from mhcflurry.numeric_proteome import ProcessingProteome, ProteinWindows, NumericCandidatePool
from mhcflurry.proteome_decoys import sample_peptide_frame_for_accessions


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (
    ["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["pad_middle", "left_pad", "right_pad", "left_pad_right_pad", "left_pad_centered_right_pad"])
@pytest.mark.parametrize("width", [11, 15])
def test_numeric_alignment_is_exactly_string_alignment(device, method, width):
    peptides = ["ACDEFGHI", "CDEFGHIKL", "DEFGHIKLMN", "EFGHIKLMNPQ"]
    options = dict(alignment_method=method, max_length=width, left_edge=3, right_edge=3)
    expected = EncodableSequences(peptides).variable_length_to_fixed_length_categorical(**options)
    numeric = NumericSequences.from_strings(peptides)
    actual = numeric.aligned_tensor(options, device)
    assert actual.dtype == torch.int8
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
    assert numeric.to_strings().tolist() == peptides


@pytest.mark.parametrize("method", ["left_pad", "right_pad"])
def test_numeric_trim_and_unknown_padding_match_strings(method):
    peptides = ["ACDEXGHIKLMNPQ", "XXACDX"]
    options = dict(alignment_method=method, max_length=8, trim=True)
    actual = NumericSequences.from_strings(peptides).aligned_tensor(options)
    expected = EncodableSequences(peptides).variable_length_to_fixed_length_categorical(**options)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_invalid_numeric_inputs_and_alignment_fail_explicitly():
    for indices, lengths in [([[21]], [1]), ([[1.0]], [1]), ([[1]], [1.5]), ([[1]], [2]), ([[1]], [-1])]:
        with pytest.raises(ValueError):
            NumericSequences(indices, lengths)
    with pytest.raises(ValueError):
        NumericSequences.from_strings(["ABCD"])
    with pytest.raises(EncodingError):
        NumericSequences.from_strings(["AAA"]).aligned_tensor({})
    with pytest.raises(NotImplementedError):
        NumericSequences.from_strings(["ACDEFGHI"]).aligned_tensor({"trim": True})
    assert NumericSequences.from_strings([]).aligned_tensor({}).shape == (0, 15)


def test_numeric_row_selection_preserves_order_and_supports_masks():
    peptides = NumericSequences.from_strings(["ACDEFGHI", "CDEFGHIKL", "DEFGHIKLMN"])
    assert peptides.take([2, 0, 2]).to_strings().tolist() == ["DEFGHIKLMN", "ACDEFGHI", "DEFGHIKLMN"]
    assert peptides.take([True, False, True]).to_strings().tolist() == ["ACDEFGHI", "DEFGHIKLMN"]
    assert len(peptides.take([])) == 0
    for rows in ([1.5], [True], [[0]]):
        with pytest.raises(ValueError):
            peptides.take(rows)


@pytest.mark.parametrize("proteins,starts,lengths", [
    ([0], [0.5], [8]), ([0], [-1], [8]), ([0], [1], [8]),
    ([1], [0], [8]), ([-1], [0], [8]), ([0], [0], [12]),
    ([0], [0], [0]), ([0, 0], [0], [8]), ([[0]], [0], [8]),
])
def test_protein_windows_reject_invalid_positions(proteins, starts, lengths):
    index = ProcessingProteome({"p": "ACDEFGHI"})
    with pytest.raises(ValueError):
        ProteinWindows(index, proteins, starts, lengths)


def test_protein_windows_validate_ownership_and_export_edges():
    index = ProcessingProteome({"p": "ACDEFGHI"})
    windows = ProteinWindows(index, [0], [0], [8])
    assert index.gather(windows).to_strings().tolist() == ["ACDEFGHI"]
    assert windows.to_frame(5).iloc[0].n_flank == "XXXXX"
    assert windows.to_frame(5).iloc[0].c_flank == "XXXXX"
    assert windows.to_frame(0).iloc[0].n_flank == ""
    assert len(ProteinWindows(index, [], [], [])) == 0
    with pytest.raises(ValueError, match="different encoded"):
        ProcessingProteome({"p": "YYYYYYYY"}).gather(windows)
    with pytest.raises(ValueError, match="Flank length"):
        windows.to_frame(-1)


@pytest.mark.parametrize("length", [2, 8, 9, 10, 11])
@pytest.mark.parametrize("seed", [1, 42])
def test_numeric_proteome_matches_previous_draws_exactly(length, seed):
    sequences = {"p1": "ACDEFGHIKLMNPQRSTVWY" * 3, "p2": "ACDEXGHIKLMNPQRSTVWY" * 2, "short": "AC"}
    accessions = ["p1", "p2", "p1", "short", None]
    excluded = {sequences["p1"][:length]}
    np.random.seed(seed)
    expected = sample_peptide_frame_for_accessions(accessions, sequences, [length], 5,
        excluded, n=75, allow_smaller=True).drop_duplicates("peptide").reset_index(drop=True)
    index = ProcessingProteome(sequences)
    batch = index.sample(accessions, length, 75, excluded, np.random.RandomState(seed))
    import pandas as pd
    pd.testing.assert_frame_equal(batch.to_frame(5), expected)
    for device in DEVICES:
        actual = index.gather(batch, device)
        assert actual.to_strings().tolist() == expected.peptide.tolist()
        np.testing.assert_array_equal(actual.indices.cpu().numpy(), index.raw_rows(batch.proteins, batch.starts, batch.lengths))


def test_candidate_pool_deduplicates_numeric_inputs_and_exports_original_rows(monkeypatch):
    import pandas as pd
    hits = pd.DataFrame(dict(peptide=["ACDEFGHI"] * 2, hit=[1, 1], protein_accession="p",
                             n_flank="XXXXX", c_flank="KLMNP", hit_id=[1, 2]))
    index = ProcessingProteome({"p": "ACDEFGHIKLMNPQRSTVWY"})
    windows = index.sample(["p"], 8, 10, set(hits.peptide), np.random.RandomState(1))
    pool = NumericCandidatePool(hits, windows, "s", 5)

    def forbidden(*args):
        raise AssertionError("Scoring materialized strings")

    monkeypatch.setattr(NumericSequences, "to_strings", forbidden)
    inputs = pool.prediction_input("cpu")
    assert len(inputs) == len(windows) + 1
    values = np.arange(len(inputs), dtype=float) + 10
    frame = pool.with_scores(values)
    assert len(frame) == len(windows) + 2
    assert frame.affinity_prediction.iloc[:2].tolist() == [10, 10]
    assert frame.sample_id.eq("s").all()


@pytest.mark.parametrize("family,merged", [("pan", False), ("pan", True), ("specific", False), ("mixed", True)])
@pytest.mark.parametrize("device", DEVICES)
def test_actual_affinity_ensemble_numeric_predictions_match_string_path(family, merged, device, monkeypatch):
    from mhcflurry import Class1AffinityPredictor
    from mhcflurry.class1_neural_network import Class1NeuralNetwork
    from mhcflurry.allele_encoding import AlleleEncoding
    from mhcflurry.common import configure_pytorch
    configure_pytorch(backend="gpu" if device == "cuda" else device)
    peptides = ["ACDEFGHI", "CDEFGHIKL", "DEFGHIKLMN", "EFGHIKLMNPQ"]
    allele = "HLA-A*02:01"
    sequences = {allele: "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRS"}
    encoding = AlleleEncoding([allele] * len(peptides), allele_to_sequence=sequences)
    models = []
    specific_models = []
    for seed in ((5, 6) if family == "pan" else (5, 6, 7)):
        model = Class1NeuralNetwork(max_epochs=1, validation_split=0, random_negative_rate=0,
            layer_sizes=[8], locally_connected_layers=[], dropout_probability=0,
            peptide_allele_merge_method="concatenate",
            peptide_encoding=dict(alignment_method="left_pad_centered_right_pad", max_length=15))
        specific = family == "specific" or (family == "mixed" and seed == 7)
        model.fit(peptides, [50, 100, 150, 200], allele_encoding=None if specific else encoding, seed=seed, verbose=0)
        (specific_models if specific else models).append(model)
    predictor = Class1AffinityPredictor(class1_pan_allele_models=models, allele_to_sequence=sequences,
        allele_to_allele_specific_models={allele: specific_models} if specific_models else {})
    if merged:
        assert predictor.optimize()
    numeric = NumericSequences.from_strings(peptides)

    def forbidden(*args, **kwargs):
        raise AssertionError("Numeric prediction converted peptides to strings")

    monkeypatch.setattr(NumericSequences, "to_strings", forbidden)
    for centrality in ("mean", "median", "robust_mean"):
        expected = predictor.predict(peptides, allele=allele, centrality_measure=centrality,
                                     model_kwargs={"batch_size": 2})
        actual = predictor.predict_numeric(numeric, allele, centrality_measure=centrality,
                                           model_kwargs={"batch_size": 2})
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    for bad in [NumericSequences.from_strings(["XXXAAAAA"]), NumericSequences.from_strings(["AAA"])]:
        with pytest.raises(ValueError):
            predictor.predict_numeric(bad, allele)
    with pytest.raises(ValueError):
        predictor.predict_numeric(numeric, "HLA-A*03:01")
    assert predictor.predict_numeric(NumericSequences.from_strings([]), allele).shape == (0,)


def test_decode_alphabet_covers_exactly_the_valid_index_range():
    """The decode table is sized by index range, not by the case-folded key count."""
    letters = "ACDEFGHIKLMNPQRSTVWYX"
    assert NumericSequences.from_strings([letters]).to_strings().tolist() == [letters]
    corrupted = NumericSequences.from_strings(["ACDEFGHI"])
    corrupted.indices[0, 0] = len(AMINO_ACID_INDEX) - 1
    with pytest.raises(IndexError):
        corrupted.to_strings()
