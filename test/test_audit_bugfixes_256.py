"""Regression tests for 2.5.6 core, processing, calibration, and FASTA bugfixes."""

from pathlib import Path
from types import SimpleNamespace

import numpy
import pandas
import pytest

from mhcflurry import (
    Class1AffinityPredictor,
    Class1NeuralNetwork,
    Class1PresentationPredictor,
    Class1ProcessingPredictor,
)
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_processing_neural_network import Class1ProcessingNeuralNetwork
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.ensemble_centrality import robust_mean
from mhcflurry.fasta import FastaParser, read_fasta_to_dataframe
from mhcflurry.flanking_encoding import FlankingEncoding
from mhcflurry.histogram_percent_rank_transform import HistogramPercentRankTransform
from mhcflurry.numeric_proteome import ProcessingProteome
from mhcflurry.numeric_sequences import NumericSequences
from mhcflurry.percentile_calibration import fit_percent_rank_transform
from mhcflurry.processing_preparation import (
    NumericCandidatePool,
    ScoredValues,
    sample_preparation_steps,
)


def _fit_tiny_pan_model(a2s, **overrides):
    hp = dict(
        max_epochs=1,
        layer_sizes=[4],
        locally_connected_layers=[],
        peptide_allele_merge_method="concatenate",
        validation_split=0.0,
        random_negative_rate=0.0,
        random_negative_constant=0,
        minibatch_size=2,
    )
    hp.update(overrides)
    m = Class1NeuralNetwork(**hp)
    enc = AlleleEncoding(["HLA-A*02:01", "HLA-A*02:01"], allele_to_sequence=a2s)
    m.fit(["SIINFEKL", "AAADDDCCC"], [50.0, 5000.0], allele_encoding=enc, verbose=0)
    return m


def test_564_batch_norm_early_width_inference_and_merge_reload():
    a2s = {"HLA-A*02:01": "A" * 34}
    m1 = _fit_tiny_pan_model(a2s, layer_sizes=[], batch_normalization=True)
    m2 = _fit_tiny_pan_model(a2s, layer_sizes=[], batch_normalization=True)
    merged = Class1NeuralNetwork.merge([m1, m2], merge_method="concatenate")
    reloaded = Class1NeuralNetwork.from_config(
        merged.get_config(), weights=merged.get_weights()
    )
    net = reloaded.network()
    assert net is not None
    enc = AlleleEncoding(["HLA-A*02:01"], allele_to_sequence=a2s)
    out = reloaded.predict(["SIINFEKL"], allele_encoding=enc, output_index=None)
    assert out.shape == (1, 2)


def test_565_merge_rejects_mismatched_encoding_hyperparameters():
    a2s = {"HLA-A*02:01": "A" * 37}
    m1 = _fit_tiny_pan_model(a2s, allele_amino_acid_encoding="BLOSUM62")
    m2 = _fit_tiny_pan_model(a2s, allele_amino_acid_encoding="one-hot")
    with pytest.raises(NotImplementedError, match="allele_amino_acid_encoding"):
        Class1NeuralNetwork.merge([m1, m2], merge_method="concatenate")

    pred = Class1AffinityPredictor(
        class1_pan_allele_models=[m1, m2], allele_to_sequence=a2s
    )
    assert pred.optimize() is False
    out = pred.predict(["SIINFEKL"], allele="HLA-A*02:01")
    assert numpy.isfinite(out).all()


def test_566_robust_mean_handles_2_and_3_model_rows_in_wide_arrays():
    arr = numpy.array([
        [1.0, 5.0, numpy.nan, numpy.nan],
        [1.0, 2.0, 10.0, numpy.nan],
        [1.0, 2.0, 3.0, 4.0],
        [numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    ])
    res = robust_mean(arr)
    assert res[0] == pytest.approx(3.0)
    assert res[1] == pytest.approx(13.0 / 3.0)
    assert res[2] == pytest.approx(2.5)
    assert numpy.isnan(res[3])


def test_567_percentile_rank_canonical_keys_and_missing_alleles():
    a2s = {"HLA-A*02:01": "A" * 37}
    m = _fit_tiny_pan_model(a2s)
    pred = Class1AffinityPredictor(
        class1_pan_allele_models=[m], allele_to_sequence=a2s
    )
    peps = [f"SIINFEK{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
    pred.calibrate_percentile_ranks(peptides=peps, alleles=["A0201"])
    assert "HLA-A*02:01" in pred.allele_to_percent_rank_transform
    ranks = pred.percentile_ranks([100.0], allele="A0201")
    assert numpy.isfinite(ranks[0])
    ranks2 = pred.percentile_ranks([100.0], allele="HLA-A*02:01")
    assert numpy.allclose(ranks, ranks2)

    with pytest.raises(ValueError, match="Missing"):
        pred.percentile_ranks([100.0, 200.0], alleles=["HLA-A*02:01", None], throw=True)
    with pytest.warns(UserWarning):
        out = pred.percentile_ranks(
            [100.0, 200.0], alleles=["HLA-A*02:01", None], throw=False
        )
    assert numpy.isfinite(out[0]) and numpy.isnan(out[1])


def test_568_empty_peptide_inputs():
    a2s = {"HLA-A*02:01": "A" * 37}
    m = _fit_tiny_pan_model(a2s)
    enc = AlleleEncoding([], allele_to_sequence=a2s)
    assert m.predict([], allele_encoding=enc).shape == (0,)
    assert m.predict([], allele_encoding=enc, output_index=None).shape == (0, 1)

    pred = Class1AffinityPredictor(
        class1_pan_allele_models=[m], allele_to_sequence=a2s
    )
    peps = [f"SIINFEK{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
    pred.calibrate_percentile_ranks(peptides=peps, alleles=["HLA-A*02:01"])
    df = pred.predict_to_dataframe(
        [],
        allele="HLA-A*02:01",
        include_percentile_ranks=True,
        include_confidence_intervals=False,
    )
    assert list(df.columns) == ["peptide", "allele", "prediction", "prediction_percentile"]


def test_569_pad_middle_zero_edges():
    arr = EncodableSequences.sequences_to_fixed_length_index_encoded_array(
        ["ACDEFGHI"],
        alignment_method="pad_middle",
        left_edge=4,
        right_edge=0,
        max_length=15,
    )
    assert arr.shape == (1, 15)

    aligned = NumericSequences.from_strings([""]).aligned_tensor({
        "alignment_method": "pad_middle",
        "left_edge": 0,
        "right_edge": 0,
        "max_length": 5,
    })
    assert aligned.shape == (1, 5)


def test_572_processing_set_weights_list_roundtrip_with_post_conv():
    net = Class1ProcessingNeuralNetwork(
        convolutional_filters=8,
        post_convolutional_dense_layer_sizes=[4],
        n_flank_length=5,
        c_flank_length=5,
    )
    m = net.make_network(
        **net.network_hyperparameter_defaults.subselect(net.hyperparameters)
    )
    orig = [w.copy() for w in m.get_weights_list()]
    m.set_weights_list(orig)  # default auto_convert_keras=True must roundtrip PyTorch weights
    after = m.get_weights_list()
    for a, b in zip(orig, after):
        assert numpy.allclose(a, b)


def test_575_predict_sequences_without_flanks_only_and_invalid_comparison_quantity():
    a2s = {"HLA-A*02:01": "A" * 37}
    aff = Class1AffinityPredictor(
        class1_pan_allele_models=[_fit_tiny_pan_model(a2s)],
        allele_to_sequence=a2s,
    )
    proc_no_flanks = Class1ProcessingPredictor(
        models=[
            Class1ProcessingNeuralNetwork(
                convolutional_filters=4,
                n_flank_length=0,
                c_flank_length=0,
                max_epochs=1,
                validation_split=0.0,
            )
        ]
    )
    proc_no_flanks.models[0].fit(
        FlankingEncoding(["SIINFEKL", "AAADDDCCC"], ["", ""], ["", ""]),
        [1.0, 0.0],
        verbose=-1,
        seed=42,
    )
    pres_no_flanks = Class1PresentationPredictor(
        affinity_predictor=aff,
        processing_predictor_with_flanks=None,
        processing_predictor_without_flanks=proc_no_flanks,
    )
    df = pres_no_flanks.predict_sequences(
        sequences={"p1": "SIINFEKLAAADDD"},
        alleles=["HLA-A*02:01"],
        include_affinity_percentile=False,
        verbose=0,
    )
    assert len(df) > 0
    assert "processing_score" in df.columns

    with pytest.raises(ValueError, match="include_affinity_percentile"):
        pres_no_flanks.predict_sequences(
            sequences={"p1": "SIINFEKLAAADDD"},
            alleles=["HLA-A*02:01"],
            comparison_quantity="affinity_percentile",
            include_affinity_percentile=False,
            verbose=0,
        )
    with pytest.raises(ValueError, match="requires alleles"):
        pres_no_flanks.predict_sequences(
            sequences={"p1": "SIINFEKLAAADDD"},
            alleles=[],
            comparison_quantity="affinity",
            verbose=0,
        )


def test_577_calibration_edge_cases():
    proc = Class1ProcessingPredictor(models=[])
    with pytest.raises(ValueError, match="constant"):
        proc.calibrate_percentile_ranks([0.5, 0.5, 0.5], method="histogram")
    with pytest.raises(ValueError, match="2 distinct bin edges"):
        HistogramPercentRankTransform().fit([0.5, 0.5], bins=[0.5])

    # Collapsed logit split: 999 values near 0.0 that collapse in probability_logits
    # plus 2 distinct high values that could land in the 20% validation split.
    scores = numpy.concatenate([
        numpy.zeros(900),
        numpy.full(98, numpy.nextafter(0.0, 1.0)),
        numpy.array([0.5, 0.9]),
    ])
    transform = fit_percent_rank_transform(
        scores, method="compact", score_transform="logit", survival=True, max_knots=128
    )
    assert transform is not None


def test_578_processing_proteome_noncanonical_excluded_and_expansion_dedup(tmp_path):
    prot = ProcessingProteome({"P1": "SIINFEKLAAAAAAAAA"})
    wins = prot.sample(
        ["P1"],
        length=8,
        n=5,
        excluded=["SIINFEKU", "SIINFEKB"],
        rng=numpy.random.default_rng(0),
    )
    assert len(wins) > 0

    # NumericCandidatePool expansion must deduplicate duplicate peptides within a round.
    hits = pandas.DataFrame({
        "peptide": ["SIINFEKL"],
        "sample_id": ["s1"],
        "hla": ["HLA-A*02:01"],
        "hit": [1],
        "peptide_len": [8],
        "n_flank": [""],
        "c_flank": [""],
        "protein_accession": ["P1"],
    })
    init_df = pandas.concat([
        hits,
        pandas.DataFrame({
            "peptide": ["AAAAAAAA"],
            "sample_id": ["s1"],
            "hla": ["HLA-A*02:01"],
            "hit": [0],
            "peptide_len": [8],
            "n_flank": [""],
            "c_flank": [""],
            "protein_accession": ["P1"],
        }),
    ], ignore_index=True)
    dup_prot = ProcessingProteome({"P1": "GGGGGGGGGGGGGG"})
    dup_wins = dup_prot.sample(["P1"], length=8, n=2, excluded=[], rng=numpy.random.default_rng(0))
    from mhcflurry.numeric_proteome import ProteinWindows

    two_dups = ProteinWindows.concatenate([dup_wins, dup_wins])
    empty_hits = hits.iloc[0:0].copy()

    def extra_cb(length, excluded, round_seed):
        return NumericCandidatePool(empty_hits, two_dups, "s1", 5)

    (tmp_path / "out.csv.matching").mkdir()
    args = SimpleNamespace(
        out=str(tmp_path / "out.csv"),
        max_expansion_rounds=1,
        decoys_per_hit=1,
        max_affinity_distance=0.05,
        matching_reference="affinity",
        random_seed=42,
    )
    gen = sample_preparation_steps(
        args=args,
        sample="s1",
        hits=hits,
        seed=42,
        initial_pool=lambda: init_df,
        additional_candidates=extra_cb,
    )
    first_req = next(gen)
    assert len(first_req) == 2
    # Send mismatched affinities so hit cannot match initial decoy -> triggers expansion round!
    unscored = gen.send(
        ScoredValues(values=numpy.array([10.0, 50000.0]), seconds=0.01)
    )
    assert isinstance(unscored, NumericCandidatePool)
    assert len(unscored.windows) == 1  # deduplicated from 2 identical windows!


def test_579_fasta_parser_reuse_utf8_and_pathlib(tmp_path):
    fa1 = tmp_path / "a.fasta"
    fa2 = tmp_path / "b.fasta"
    fa1.write_text(">seq1 α-chain description\nAAAA\n", encoding="utf-8")
    fa2.write_text(">seq2 β-chain description\nBBBB\n", encoding="utf-8")

    parser = FastaParser()
    r1 = list(parser.iterate_over_file(fa1, full_descriptions=True))
    r2 = list(parser.iterate_over_file(fa2, full_descriptions=True))
    assert r1 == [("seq1 α-chain description", "AAAA")]
    assert r2 == [("seq2 β-chain description", "BBBB")]

    df = read_fasta_to_dataframe(Path(fa1))
    assert list(df["sequence_id"]) == ["seq1"]
    assert list(df["sequence"]) == ["AAAA"]
