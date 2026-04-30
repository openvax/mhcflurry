"""Parity + perf tests for ``Class1AffinityPredictor.calibrate_percentile_ranks_fast``.

The fast path batches many alleles into a single forward after caching the
peptide-side activations per network. The parity test anchors that the
alternative schedule produces the same ``PercentRankTransform`` as the
slow legacy path; the smoke test anchors that it actually runs end to
end against a downloaded pan-allele release.
"""

import torch

import numpy
import pandas
import pytest

from mhcflurry import Class1AffinityPredictor
from mhcflurry.common import positional_frequency_matrix, random_peptides
from mhcflurry.downloads import get_path
from mhcflurry.encodable_sequences import EncodableSequences


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


def _legacy_motif_summary_from_predictions(
        peptides, predictions, alleles, summary_top_peptide_fractions):
    frequency_matrices = []
    length_distributions = []
    for allele_i, allele in enumerate(alleles):
        df = pandas.DataFrame({
            "peptide": peptides,
            "prediction": predictions[allele_i],
        }).drop_duplicates("peptide").set_index("peptide")
        df["length"] = df.index.str.len()
        for (length, sub_df) in df.groupby("length"):
            for cutoff_fraction in summary_top_peptide_fractions:
                selected = sub_df.prediction.nsmallest(
                    max(int(len(sub_df) * cutoff_fraction), 1),
                ).index.values
                matrix = positional_frequency_matrix(selected).reset_index()
                original_columns = list(matrix.columns)
                matrix["allele"] = allele
                matrix["length"] = length
                matrix["cutoff_fraction"] = cutoff_fraction
                matrix["cutoff_count"] = len(selected)
                matrix = matrix[
                    ["allele", "length", "cutoff_fraction", "cutoff_count"]
                    + original_columns
                ]
                frequency_matrices.append(matrix)

        for cutoff_fraction in summary_top_peptide_fractions:
            cutoff_count = max(int(len(df) * cutoff_fraction), 1)
            length_distribution = df.prediction.nsmallest(
                cutoff_count,
            ).index.str.len().value_counts()
            length_distribution.index.name = "length"
            length_distribution /= length_distribution.sum()
            length_distribution = length_distribution.to_frame("fraction")
            length_distribution = length_distribution.reset_index()
            length_distribution["allele"] = allele
            length_distribution["cutoff_fraction"] = cutoff_fraction
            length_distribution["cutoff_count"] = cutoff_count
            length_distribution = length_distribution[[
                "allele",
                "cutoff_fraction",
                "cutoff_count",
                "length",
                "fraction",
            ]].sort_values(["cutoff_fraction", "length"])
            length_distributions.append(length_distribution)

    return {
        "frequency_matrices": pandas.concat(
            frequency_matrices, ignore_index=True,
        ),
        "length_distributions": pandas.concat(
            length_distributions, ignore_index=True,
        ),
    }


def test_motif_summary_gpu_helper_matches_legacy_edge_cases():
    peptides = [
        "AA",
        "CC",
        "DD",
        "EE",
        "FF",
        "AZ",  # Z should be treated like X: dropped from the 20 AA columns.
        "GGG",
        "HHH",
        "GGG",  # Duplicate should keep the first prediction, not this one.
        "ACDEFGHIKLMNPQRS",  # Observed length outside the default 8-15 range.
        "AZDEFGHIKLMNPQRS",
    ]
    alleles = ["allele-a", "allele-b"]
    predictions = numpy.asarray([
        [1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 9.0, 1.0, 0.0, 2.0, 3.0],
        [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 0.0, 5.0, 0.0, 1.0, 1.0],
    ])
    cutoff_fractions = (0.4, 1.0)

    state = Class1AffinityPredictor._prepare_motif_summary_state_gpu(
        EncodableSequences.create(peptides),
        torch.device("cpu"),
    )
    fast_freq, fast_ld = Class1AffinityPredictor._motif_summary_chunk_gpu(
        torch.from_numpy(predictions),
        state,
        cutoff_fractions,
        alleles,
    )
    fast_summary = {
        "frequency_matrices": pandas.concat(fast_freq, ignore_index=True),
        "length_distributions": pandas.concat(fast_ld, ignore_index=True),
    }
    legacy_summary = _legacy_motif_summary_from_predictions(
        peptides,
        predictions,
        alleles,
        cutoff_fractions,
    )

    freq_sort = ["allele", "length", "cutoff_fraction", "position"]
    pandas.testing.assert_frame_equal(
        legacy_summary["frequency_matrices"].sort_values(
            freq_sort,
        ).reset_index(drop=True),
        fast_summary["frequency_matrices"].sort_values(
            freq_sort,
        ).reset_index(drop=True),
        check_exact=False,
        rtol=0,
        atol=0,
    )
    ld_sort = ["allele", "cutoff_fraction", "length"]
    pandas.testing.assert_frame_equal(
        legacy_summary["length_distributions"].sort_values(
            ld_sort,
        ).reset_index(drop=True),
        fast_summary["length_distributions"].sort_values(
            ld_sort,
        ).reset_index(drop=True),
        check_exact=False,
        rtol=0,
        atol=0,
    )


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
    # frequency_matrices: per-allele content must match the legacy slow
    # path. A separate synthetic test covers exact top-k tie behavior;
    # this end-to-end test keeps a small tolerance for numerical drift
    # near a real model's cutoff boundary.
    lf = legacy_summary["frequency_matrices"]
    ff = fast_summary["frequency_matrices"]
    assert set(lf.allele.unique()) == set(ff.allele.unique())
    aa_columns = [c for c in lf.columns if len(c) == 1 and c.isalpha()]
    assert "A" in aa_columns and "Y" in aa_columns
    # Sanity: rows must sum close to 1 within each position (no X in
    # random_peptides, so should be exactly 1.0 modulo float rounding).
    fast_row_sums = ff[aa_columns].sum(axis=1)
    numpy.testing.assert_allclose(fast_row_sums, 1.0, rtol=0, atol=1e-9)
    # Sanity: different alleles must produce different motif matrices.
    fast_allele_signatures = ff.groupby("allele")[aa_columns].mean()
    assert fast_allele_signatures.nunique(axis=0).max() > 1, (
        "all alleles produced identical motif matrices — per-allele "
        "topk is not selecting different peptide sets per row"
    )
    for (allele, length, cutoff), sub_l in lf.groupby(
        ["allele", "length", "cutoff_fraction"]
    ):
        sub_f = ff[
            (ff.allele == allele)
            & (ff.length == length)
            & (ff.cutoff_fraction == cutoff)
        ]
        assert len(sub_l) == len(sub_f)
        sub_l_sorted = sub_l.sort_values("position").reset_index(drop=True)
        sub_f_sorted = sub_f.sort_values("position").reset_index(drop=True)
        k = int(sub_l_sorted["cutoff_count"].iloc[0])
        # Per-cell drift above one peptide's contribution would mean
        # the per-allele top-k selection diverged materially.
        diff = (
            sub_l_sorted[aa_columns].to_numpy()
            - sub_f_sorted[aa_columns].to_numpy()
        )
        max_abs = numpy.abs(diff).max()
        assert max_abs <= 1.0 / k + 1e-9, (
            f"frequency-matrix drift {max_abs:.4g} exceeds 1/k = "
            f"{1.0 / k:.4g} for ({allele}, length={length}, "
            f"cutoff={cutoff}) — per-allele top-k diverged beyond ties"
        )


def test_check_training_batch_fits_shrinks_loudly_on_oom(caplog):
    """``check_training_batch_fits`` shrinks an oversized batch and logs
    a loud warning explaining the training-dynamics drift."""
    import logging

    from mhcflurry.class1_neural_network import (
        check_training_batch_fits,
        _TRAINING_PEAK_MULTIPLIER,
        _estimate_peak_bytes_per_row,
    )

    class FakeCUDA:
        """Pretend we're on CUDA with 8 GB free, 2 workers per GPU."""
        def __init__(self):
            self.type = "cuda"
        def __str__(self):
            return "cuda:0"

    # Swap out _free_device_memory_bytes for the duration of this test.
    from mhcflurry import class1_neural_network as cnn
    saved = cnn._free_device_memory_bytes
    try:
        cnn._free_device_memory_bytes = lambda device: 8 * (1 << 30)

        class TinyModel:
            # Pretend peak row = 1 KB; 8 GB / 2 workers / 2 (0.5 fraction)
            # / (1024 * 4) = 1 M rows max — so 500k fits, 4M doesn't.
            peptide_encoding_shape = (15, 21)
            lc_layers = []
            peptide_dense_layers = []
            allele_embedding = None
            allele_dense_layers = []
            class _Layer:
                out_features = 128
            dense_layers = [_Layer]

        model = TinyModel()
        peak = _estimate_peak_bytes_per_row(model) * _TRAINING_PEAK_MULTIPLIER
        assert peak > 0
        device = FakeCUDA()

        # Small batch — fits easily, no shrink, no warning.
        with caplog.at_level(logging.WARNING, logger="root"):
            fits, shrunk = check_training_batch_fits(
                128, device, model, num_workers_per_gpu=2,
            )
        assert not shrunk
        assert fits == 128

        # Huge batch — definitely exceeds budget, must shrink + warn.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="root"):
            fits, shrunk = check_training_batch_fits(
                10_000_000, device, model, num_workers_per_gpu=2,
            )
        assert shrunk
        assert 64 <= fits < 10_000_000
        # Power-of-two floor.
        assert fits & (fits - 1) == 0
        joined = " ".join(r.message for r in caplog.records)
        assert "TRAINING BATCH WILL NOT FIT" in joined
        assert "CHANGES TRAINING DYNAMICS" in joined
    finally:
        cnn._free_device_memory_bytes = saved


def test_fit_end_to_end_shrinks_minibatch_when_vram_too_small(caplog):
    """End-to-end: a tiny fit() on a mocked small-VRAM device should
    invoke ``check_training_batch_fits``, shrink the configured
    minibatch, record it in fit_info, and *not* mutate the predictor's
    saved hyperparameters dict.

    Exercises MPS (the only non-CPU device reliably available in the
    test env); MPS's guard path is identical to CUDA's, modulo the
    dtype fallback tested elsewhere.
    """
    import logging
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available — guard end-to-end test needs a non-CPU device")

    from mhcflurry import class1_neural_network as cnn
    from mhcflurry.class1_neural_network import Class1NeuralNetwork
    from mhcflurry.common import random_peptides

    # Override MPS free-memory to be tiny so the shrink fires.
    saved_free = cnn._free_device_memory_bytes
    try:
        cnn._free_device_memory_bytes = lambda device: (
            128 * (1 << 20) if device.type == "mps" else saved_free(device)
        )

        hyperparameters = dict(
            activation="tanh",
            layer_sizes=[16],
            max_epochs=1,
            early_stopping=False,
            validation_split=0.0,
            locally_connected_layers=[],
            dense_layer_l1_regularization=0.0,
            dropout_probability=0.0,
            minibatch_size=524288,  # absurdly big for 128MB — must shrink
        )

        peptides = random_peptides(128, length=9)
        affinities = numpy.random.uniform(10, 50000, 128)

        predictor = Class1NeuralNetwork(**hyperparameters)
        with caplog.at_level(logging.WARNING, logger="root"):
            try:
                predictor.fit(peptides, affinities, verbose=0)
            except Exception:
                # Downstream eagerness may fail on the tiny harness;
                # the guard runs BEFORE any training work so fit_info
                # captures it regardless.
                pass

        joined = " ".join(r.message for r in caplog.records)
        assert "TRAINING BATCH WILL NOT FIT" in joined, (
            "expected the loud shrink warning in the log. Captured: "
            + joined[:400]
        )
        assert predictor.fit_info, "fit_info should be populated even on early failure"
        info = predictor.fit_info[-1]
        assert info.get("minibatch_size_shrunk_from") == 524288
        effective = info.get("minibatch_size_shrunk_to")
        assert effective is not None and 64 <= effective < 524288, (
            "shrunk minibatch must be between the floor (64) and the "
            "requested size, got %r" % effective
        )
        assert info.get("effective_minibatch_size") == effective
        # Hyperparameters dict must NOT be mutated — the saved config
        # preserves the user's original intent. This is the fix for
        # the code/comment mismatch called out in the code review.
        assert predictor.hyperparameters["minibatch_size"] == 524288, (
            "hyperparameters were mutated during fit(); the saved model "
            "config would no longer reflect the user's configured value"
        )
    finally:
        cnn._free_device_memory_bytes = saved_free


def test_processing_nn_auto_batch_matches_explicit_size():
    """``Class1ProcessingNeuralNetwork.predict_encoded`` with
    ``batch_size="auto"`` must produce bit-identical predictions to a
    pinned integer batch. The auto-size resolves device+model at call
    time; the only thing that should change is how many rows go
    through each forward, not the output.
    """
    try:
        from mhcflurry.downloads import get_path
        from mhcflurry import Class1ProcessingPredictor
    except Exception as exc:
        pytest.skip(f"mhcflurry processing predictor imports unavailable: {exc}")
    # Public release ships multiple flank-mode variants under
    # models_class1_processing/; pick the with-flanks flavor since
    # the test passes explicit flanks.
    try:
        models_dir = get_path(
            "models_class1_processing", "models.selected.with_flanks",
        )
    except Exception as exc:
        pytest.skip(f"public processing predictor not available: {exc}")

    processing = Class1ProcessingPredictor.load(models_dir)
    # Small dataset — correctness check, not perf.
    peptides = [
        "SIINFEKL", "GILGFVFTL", "SLYNTVATL", "NLVPMVATV",
        "ELAGIGILTV", "KVAELVHFL", "TLDSQVMSL", "YVDPVITSI",
    ]
    n_flanks = ["" for _ in peptides]
    c_flanks = ["" for _ in peptides]

    # Pinned batch
    pinned = processing.predict(
        peptides=peptides, n_flanks=n_flanks, c_flanks=c_flanks,
        batch_size=2,
    )
    # Auto batch
    auto = processing.predict(
        peptides=peptides, n_flanks=n_flanks, c_flanks=c_flanks,
        batch_size="auto",
    )
    numpy.testing.assert_allclose(
        pinned, auto, rtol=0, atol=1e-6,
        err_msg="auto-sized processing predict must match pinned-size output",
    )


def test_compute_prediction_batch_size_scales_with_memory_and_workers():
    """``compute_prediction_batch_size`` respects free VRAM and the
    workers-per-GPU partition."""
    from mhcflurry.class1_neural_network import (
        compute_prediction_batch_size,
        _AUTO_BATCH_CPU_FALLBACK,
        _AUTO_BATCH_MAX_ROWS,
        _AUTO_BATCH_MIN_ROWS,
    )

    cpu = torch.device("cpu")
    # CPU short-circuit: fixed fallback regardless of workers / memory.
    # Large batches on CPU thrash L3 and don't help the small networks
    # mhcflurry trains.
    assert compute_prediction_batch_size(cpu) == _AUTO_BATCH_CPU_FALLBACK
    assert compute_prediction_batch_size(
        cpu, num_workers_per_gpu=16,
    ) == _AUTO_BATCH_CPU_FALLBACK

    # Model=None (fallback estimate) still produces a within-bounds
    # batch on MPS/CUDA if either is available locally.
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        dev = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps")
        )
        single = compute_prediction_batch_size(dev)
        shared = compute_prediction_batch_size(dev, num_workers_per_gpu=8)
        assert _AUTO_BATCH_MIN_ROWS <= single <= _AUTO_BATCH_MAX_ROWS
        assert _AUTO_BATCH_MIN_ROWS <= shared <= _AUTO_BATCH_MAX_ROWS
        # Sharing the GPU with 8 workers partitions the budget — each
        # worker's batch is no larger than the unshared case. Equality
        # is allowed because the min_rows floor clamps both cases
        # when free memory is tight.
        assert shared <= single


def test_peptide_sequences_fingerprint_distinguishes_middle_changes():
    """Same length, first, last → different middle must yield different keys.

    Regression for the legacy ``(count, first, last)`` cache_signature
    which collided whenever the middle peptides differed but the
    boundary peptides matched. Reusing a stale peptide-stage cache in
    that case silently produced wrong PercentRankTransforms.
    """
    from mhcflurry.class1_affinity_predictor import (
        _peptide_sequences_fingerprint,
    )

    a = _peptide_sequences_fingerprint(["AAAA", "BBBB", "CCCC"])
    b = _peptide_sequences_fingerprint(["AAAA", "XXXX", "CCCC"])
    assert a != b


def test_peptide_sequences_fingerprint_length_prefix_prevents_concat_collision():
    """Length prefixing prevents the trivial concatenation collision."""
    from mhcflurry.class1_affinity_predictor import (
        _peptide_sequences_fingerprint,
    )

    assert (
        _peptide_sequences_fingerprint(["AB", "C"])
        != _peptide_sequences_fingerprint(["A", "BC"])
    )


def test_peptide_sequences_fingerprint_order_sensitive():
    """Reordering the same peptides must change the fingerprint."""
    from mhcflurry.class1_affinity_predictor import (
        _peptide_sequences_fingerprint,
    )

    assert (
        _peptide_sequences_fingerprint(["A", "B"])
        != _peptide_sequences_fingerprint(["B", "A"])
    )


def test_calibration_fast_cache_state_lifecycle():
    """``_calibration_fast_cache`` is lazy and ``clear`` releases it."""
    from mhcflurry.class1_affinity_predictor import (
        Class1AffinityPredictor,
        _CalibrationFastCache,
    )

    predictor = Class1AffinityPredictor()
    assert getattr(predictor, "_calibration_fast_cache_state", None) is None

    cache = predictor._calibration_fast_cache()
    assert isinstance(cache, _CalibrationFastCache)
    assert predictor._calibration_fast_cache() is cache  # idempotent

    cache.cached_stages = ["sentinel"]
    cache.stage_signature = ("sig",)
    predictor.clear_calibration_fast_cache()
    assert getattr(predictor, "_calibration_fast_cache_state", None) is None
    # clearing again is a no-op
    predictor.clear_calibration_fast_cache()


def test_calibrate_fast_rejects_scalar_bins_before_model_work():
    predictor = Class1AffinityPredictor(
        class1_pan_allele_models=[object()],
        allele_to_sequence={"A": "AAAAAAAA"},
    )
    with pytest.raises(ValueError, match="requires explicit IC50 bin edges"):
        predictor.calibrate_percentile_ranks_fast(
            peptides=["AAAA"],
            alleles=["A"],
            bins=3,
            peptide_batch_size=1,
            allele_batch_size=1,
            device="cpu",
        )


def test_calibrate_fast_cache_signature_cleared_if_rebuild_fails():
    class FailingModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    class FailingNetwork:
        def allele_encoding_to_network_input(self, allele_encoding):
            return None, object()

        def set_allele_representations(self, allele_representations):
            pass

        def network(self, borrow=True):
            return FailingModel()

        def peptides_to_network_input(self, encoded_peptides):
            raise RuntimeError("intentional cache rebuild failure")

        def uses_peptide_torch_encoding(self):
            return True

    predictor = Class1AffinityPredictor(
        class1_pan_allele_models=[FailingNetwork()],
        allele_to_sequence={"A": "AAAAAAAA"},
    )
    cache = predictor._calibration_fast_cache()
    cache.stage_signature = ("old-signature",)
    cache.cached_stages = ["old-cache"]

    with pytest.raises(RuntimeError, match="intentional cache rebuild failure"):
        predictor.calibrate_percentile_ranks_fast(
            peptides=["AAAA"],
            alleles=["A"],
            bins=numpy.array([0.0, 1.0, 2.0]),
            peptide_batch_size=1,
            allele_batch_size=1,
            device="cpu",
        )

    assert cache.stage_signature is None
    assert cache.cached_stages is None
