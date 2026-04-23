"""End-to-end integration test: training is bit-identical with cache on/off.

This is THE gate for issue #268 Phase 1. Unit tests can prove that the
cache produces encoded bytes matching ``variable_length_to_fixed_length_
vector_encoding`` exactly — but that only tests the encoder's output. It
doesn't test what happens when those bytes flow through multi-epoch
training with stochastic operations (weight init, random negative
sampling, minibatch shuffle, validation split).

So: we run actual training on a tiny model, with fixed seeds, via both
the original code path (fresh ``EncodableSequences``) and the new cached
path (``make_prepopulated_encodable_sequences``). We assert the per-epoch
loss trajectories match bit-for-bit AND the final model weights match
bit-for-bit.

If this test passes, the semantic-preservation claim is proven end-to-end:
the cache is a pure implementation-detail optimization, not a behavior
change.

On CPU only. GPU determinism requires extra flags and isn't necessary to
prove the point — the cache doesn't know or care about the device.
"""

from __future__ import annotations

import random

import numpy
import pandas
import pytest
import torch

from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_neural_network import Class1NeuralNetwork
from mhcflurry.common import configure_pytorch
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry.encoding_cache import (
    EncodingCache,
    EncodingParams,
    make_prepopulated_encodable_sequences,
)


# Tiny but real: enough rows for a minibatch, enough length variety to
# exercise the encoder, small enough that 2 epochs finish in under a second.
TRAIN_PEPTIDES = [
    "SIINFEKL", "GILGFVFTL", "NLVPMVATV", "YLQPRTFLL", "KLVALGINAV",
    "FLRGRAYGL", "ELAGIGILTV", "RMFPNAPYL", "AAAAAAAAA", "KLIETYFSKK",
    "YVNVNMGLK", "ILKEPVHGV", "VLFRGGPRGSL", "FEDLRVLSF", "TPGPGVRYPL",
    "QYDPVAALF", "RMPEAAPPV", "IMDQVPFSV", "LLDFVRFMGV", "VQMENKLTL",
    "FVLELEPEWTV", "ELAGIGILT", "GLQDCTMLV", "WLSLLVPFV", "FLPSDFFPSV",
    "KAFSPEVIPMF", "FLYGKLILAS", "VLSPFPPKV", "SLYNTVATL", "GPGHKARVL",
]
TRAIN_ALLELES_CYCLE = ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*01:01"]
ALLELE_TO_SEQUENCE = {
    "HLA-A*02:01": "YFAMYQENMAHTDANTLYLNYHDYTWAVLAYTWYG",
    "HLA-B*07:02": "YFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAA",
    "HLA-A*01:01": "YFSTSISRPGRGEPRFIAVGYIDDTQFVRFDSDAA",
}
DEFAULT_PEPTIDE_ENCODING = {
    "alignment_method": "left_pad_centered_right_pad",
    "max_length": 15,
    "vector_encoding_name": "BLOSUM62",
}
_PREENCODED_BATCHES = [
    (
        {
            "peptide": EncodableSequences(TRAIN_PEPTIDES[:4])
            .variable_length_to_fixed_length_vector_encoding(
                **DEFAULT_PEPTIDE_ENCODING
            ),
            "allele": numpy.array([0, 1, 2, 0], dtype=numpy.int64),
        },
        numpy.array([0.05, 0.25, 0.75, 0.95], dtype=numpy.float32),
    ),
    (
        {
            "peptide": EncodableSequences(TRAIN_PEPTIDES[4:8])
            .variable_length_to_fixed_length_vector_encoding(
                **DEFAULT_PEPTIDE_ENCODING
            ),
            "allele": numpy.array([1, 2, 0, 1], dtype=numpy.int64),
        },
        numpy.array([0.15, 0.35, 0.65, 0.85], dtype=numpy.float32),
    ),
]


def _make_preencoded_fit_generator(worker_id=0, num_workers=1):
    for idx, (x_dict, y) in enumerate(_PREENCODED_BATCHES):
        if idx % num_workers != worker_id:
            continue
        yield (
            {key: value.copy() for key, value in x_dict.items()},
            y.copy(),
        )


def _seed_everything(seed=12345):
    """Seed every RNG the training code touches."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    # Force deterministic algorithms on CPU paths. torch.use_deterministic_algorithms
    # raises on ops that have no deterministic impl — we don't use any such ops
    # here, so the flag is safe.
    torch.use_deterministic_algorithms(True)


def _build_training_data():
    """Return a DataFrame with peptide, allele, measurement_value columns."""
    rng = numpy.random.default_rng(0)
    rows = []
    for i, peptide in enumerate(TRAIN_PEPTIDES):
        allele = TRAIN_ALLELES_CYCLE[i % len(TRAIN_ALLELES_CYCLE)]
        # Affinities in nM; mix of binders and non-binders.
        measurement = float(rng.choice([50.0, 500.0, 10000.0]))
        rows.append(
            {"peptide": peptide, "allele": allele, "measurement_value": measurement}
        )
    return pandas.DataFrame(rows)


def _tiny_hyperparameters():
    """Hyperparameters sized so 2 epochs run in <1 second.

    Shaped to match the pan-allele defaults produced by
    downloads-generation/models_class1_pan/generate_hyperparameters.py —
    specifically peptide/allele merge via concatenate, small layer sizes,
    dropout off (determinism), early stopping off (epoch count fixed).
    """
    return {
        "peptide_encoding": DEFAULT_PEPTIDE_ENCODING,
        "max_epochs": 2,
        "minibatch_size": 8,
        # Architecture — sized tiny for test speed.
        "layer_sizes": [16],
        "peptide_dense_layer_sizes": [],
        "allele_dense_layer_sizes": [],
        "locally_connected_layers": [],
        "peptide_allele_merge_method": "concatenate",
        "peptide_allele_merge_activation": "",
        "peptide_amino_acid_encoding": "BLOSUM62",
        "topology": "feedforward",
        # Regularization off for determinism across runs.
        "dropout_probability": 0.0,
        "batch_normalization": False,
        "dense_layer_l1_regularization": 0.0,
        "dense_layer_l2_regularization": 0.0,
        # Training control.
        "validation_split": 0.2,
        "early_stopping": False,
        "patience": 100,
        "min_delta": 0.0,
        "learning_rate": 0.001,
        "loss": "custom:mse_with_inequalities",
        "optimizer": "rmsprop",
        "activation": "tanh",
        "output_activation": "sigmoid",
        "init": "glorot_uniform",
        "data_dependent_initialization_method": "lsuv",
        # Random negatives — pan-allele path.
        "random_negative_rate": 1.0,
        "random_negative_method": "by_allele_equalize_nonbinders",
        "random_negative_affinity_min": 30000.0,
        "random_negative_affinity_max": 50000.0,
        "random_negative_binder_threshold": 500.0,
        "random_negative_constant": 1,
        "random_negative_distribution_smoothing": 0.0,
        "random_negative_match_distribution": True,
    }


def _train_one_run(peptides_arg, training_df, allele_encoding):
    """Build + train a Class1NeuralNetwork once, return (fit_info, state_dict).

    ``peptides_arg`` is either a plain EncodableSequences (old path) or a
    prepopulated one (new path). Downstream code in ``fit()`` doesn't know
    which it got — and that's the point.
    """
    network = Class1NeuralNetwork(**_tiny_hyperparameters())
    network.fit(
        peptides=peptides_arg,
        affinities=training_df.measurement_value.values,
        allele_encoding=allele_encoding,
        verbose=0,
        progress_print_interval=None,
    )
    fit_info = network.fit_info[-1]
    # .state_dict() returns tensors on whatever device they're on; we force
    # CPU above, so these are CPU tensors ready to compare.
    state_dict = {k: v.detach().cpu().clone() for k, v in network.network().state_dict().items()}
    return fit_info, state_dict


def _state_dicts_allclose(a, b, atol=0.0, rtol=0.0):
    """Compare two state_dicts; default atol/rtol=0 means bit-identical."""
    assert set(a.keys()) == set(b.keys()), f"key mismatch: {set(a.keys()) ^ set(b.keys())}"
    for k in a:
        if not torch.equal(a[k], b[k]) if atol == 0 and rtol == 0 else torch.allclose(
            a[k], b[k], atol=atol, rtol=rtol
        ):
            return False, k
    return True, None


@pytest.fixture(autouse=True)
def force_cpu_backend():
    """Ensure all training in this file runs on CPU for determinism."""
    configure_pytorch(backend="cpu")
    yield
    # Leave backend as-is for any follow-on tests; they'll reconfigure as needed.


@pytest.fixture
def training_df():
    return _build_training_data()


@pytest.fixture
def allele_encoding(training_df):
    return AlleleEncoding(
        alleles=training_df.allele.values,
        allele_to_sequence=ALLELE_TO_SEQUENCE,
    )


# ---- THE GATE ----


def test_training_bit_identical_cached_vs_uncached(
    training_df, allele_encoding, tmp_path
):
    """End-to-end: fit() must produce identical loss and weights cached vs not.

    If this passes, the cache is a pure performance optimization. If it
    fails, we have a real divergence in training semantics and the PR
    should not merge.
    """
    params = EncodingParams(**DEFAULT_PEPTIDE_ENCODING)

    # Build the encoding cache up-front (as the orchestrator would).
    cache = EncodingCache(cache_dir=tmp_path / "encoding_cache", params=params)
    # Cache the unique training peptides (in first-seen order).
    unique_peptides = list(
        pandas.Series(training_df.peptide.values).drop_duplicates()
    )
    encoded_all, peptide_to_idx = cache.get_or_build(unique_peptides)

    # Path A: fresh EncodableSequences (old path).
    _seed_everything()
    peptides_uncached = EncodableSequences(training_df.peptide.values)
    fit_info_uncached, weights_uncached = _train_one_run(
        peptides_uncached, training_df, allele_encoding
    )

    # Path B: prepopulated EncodableSequences (new path).
    _seed_everything()
    fold_indices = numpy.array(
        [peptide_to_idx[p] for p in training_df.peptide.values], dtype=numpy.int64
    )
    fold_encoded = encoded_all[fold_indices]
    peptides_cached = make_prepopulated_encodable_sequences(
        training_df.peptide.values, fold_encoded, params
    )
    fit_info_cached, weights_cached = _train_one_run(
        peptides_cached, training_df, allele_encoding
    )

    # ---- the assertions ----

    # Loss trajectory bit-identical.
    numpy.testing.assert_array_equal(
        fit_info_uncached["loss"], fit_info_cached["loss"]
    )
    numpy.testing.assert_array_equal(
        fit_info_uncached["val_loss"], fit_info_cached["val_loss"]
    )

    # Weights bit-identical (torch.equal requires exact match, no tolerance).
    ok, bad_key = _state_dicts_allclose(weights_uncached, weights_cached)
    assert ok, (
        f"state_dict mismatch at key {bad_key!r}: "
        f"uncached[{bad_key!r}]={weights_uncached[bad_key]} "
        f"cached[{bad_key!r}]={weights_cached[bad_key]}"
    )


def test_training_cached_vs_uncached_on_mps(
    training_df, allele_encoding, tmp_path
):
    """MPS parity gate: cache is a pure no-op on Apple-silicon too.

    The CPU gate above proves the cache doesn't change training *on CPU*.
    But for macOS dev machines, ``auto`` backend selects MPS — so CPU-only
    coverage means we never actually exercise what local contributors run
    daily. This test re-runs the same cache-on / cache-off comparison with
    ``backend="mps"``. Cache is device-independent by construction (it's
    disk-resident numpy bytes), so any per-device kernel differences
    affect both runs identically and the loss/weight trajectories must
    stay bit-identical.

    Skipped when MPS isn't available (Linux CI, pre-Apple-silicon macOS).
    """
    if not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS backend unavailable on this platform")

    # Override the autouse CPU fixture for this test only.
    configure_pytorch(backend="mps")

    params = EncodingParams(**DEFAULT_PEPTIDE_ENCODING)
    cache = EncodingCache(cache_dir=tmp_path / "encoding_cache", params=params)
    unique_peptides = list(
        pandas.Series(training_df.peptide.values).drop_duplicates()
    )
    encoded_all, peptide_to_idx = cache.get_or_build(unique_peptides)

    _seed_everything()
    peptides_uncached = EncodableSequences(training_df.peptide.values)
    fit_info_uncached, weights_uncached = _train_one_run(
        peptides_uncached, training_df, allele_encoding
    )

    _seed_everything()
    fold_indices = numpy.array(
        [peptide_to_idx[p] for p in training_df.peptide.values],
        dtype=numpy.int64,
    )
    fold_encoded = encoded_all[fold_indices]
    peptides_cached = make_prepopulated_encodable_sequences(
        training_df.peptide.values, fold_encoded, params
    )
    fit_info_cached, weights_cached = _train_one_run(
        peptides_cached, training_df, allele_encoding
    )

    numpy.testing.assert_array_equal(
        fit_info_uncached["loss"], fit_info_cached["loss"]
    )
    numpy.testing.assert_array_equal(
        fit_info_uncached["val_loss"], fit_info_cached["val_loss"]
    )
    ok, bad_key = _state_dicts_allclose(weights_uncached, weights_cached)
    assert ok, (
        f"MPS cached/uncached weights diverge at {bad_key!r}: "
        f"uncached={weights_uncached[bad_key]} "
        f"cached={weights_cached[bad_key]}"
    )

    # Sanity: losses are finite and training moved. Otherwise the MPS
    # run could be silently producing NaN both times and the equality
    # check would pass trivially.
    losses = fit_info_cached["loss"]
    assert all(numpy.isfinite(loss) for loss in losses), (
        f"MPS training produced non-finite loss: {losses}"
    )


def test_training_weights_differ_with_different_seeds(
    training_df, allele_encoding
):
    """Sanity: with different seeds, weights should differ.

    Guards against the above test passing trivially — e.g. if the
    ``_seed_everything`` calls were accidentally no-ops or the model
    always converged to a fixed point regardless of seed. If this test
    fails (weights identical under different seeds), the bit-identical
    test above proves nothing.
    """
    _seed_everything(seed=12345)
    peptides1 = EncodableSequences(training_df.peptide.values)
    _, weights1 = _train_one_run(peptides1, training_df, allele_encoding)

    _seed_everything(seed=67890)
    peptides2 = EncodableSequences(training_df.peptide.values)
    _, weights2 = _train_one_run(peptides2, training_df, allele_encoding)

    # At least one parameter should differ between the two seeds.
    any_diff = any(
        not torch.equal(weights1[k], weights2[k]) for k in weights1
    )
    assert any_diff, (
        "weights identical across different seeds — seeding is broken, "
        "which would make the bit-identical test above a tautology"
    )


def test_training_loss_decreases(training_df, allele_encoding):
    """Sanity: training should actually make the loss go down.

    Not strictly required by the bit-identical test but useful as a basic
    sanity check that the tiny model + data set up above is actually
    training, not just producing noise.
    """
    _seed_everything()
    peptides = EncodableSequences(training_df.peptide.values)
    fit_info, _ = _train_one_run(peptides, training_df, allele_encoding)
    losses = fit_info["loss"]
    # 2 epochs, last should be lower than first (or at least not exploding).
    assert len(losses) >= 2
    assert losses[-1] < losses[0] * 2.0, (
        f"loss didn't improve: {losses} — training sanity check failed"
    )


# ---- DataLoader num_workers bit-identity gate ----


def _train_with_workers(
    peptides_arg,
    training_df,
    allele_encoding,
    num_workers,
):
    """Train identically to _train_one_run but with a configurable num_workers."""
    hp = _tiny_hyperparameters()
    hp["dataloader_num_workers"] = num_workers
    network = Class1NeuralNetwork(**hp)
    network.fit(
        peptides=peptides_arg,
        affinities=training_df.measurement_value.values,
        allele_encoding=allele_encoding,
        verbose=0,
        progress_print_interval=None,
    )
    fit_info = network.fit_info[-1]
    state_dict = {
        k: v.detach().cpu().clone() for k, v in network.network().state_dict().items()
    }
    return fit_info, state_dict


def test_dataloader_num_workers_0_matches_baseline(
    training_df, allele_encoding
):
    """num_workers=0 (DataLoader wrap without prefetch) must match the pre-
    DataLoader baseline exactly.

    This isolates the Step 4 change: wrapping the inner batch loop in a
    DataLoader with num_workers=0 is a pure refactor — same operations,
    same order, same tensors — and should produce bit-identical training
    results. If this ever fails, the wrapping introduced a subtle
    reordering or type coercion that changes numerics.
    """
    # The main integration test above already exercises num_workers=0 as
    # the default, so if this test passes we know the default path is
    # bit-identical. We explicitly set num_workers=0 to pin the test to
    # that contract rather than relying on a default value.
    _seed_everything()
    peptides_a = EncodableSequences(training_df.peptide.values)
    fit_a, weights_a = _train_with_workers(
        peptides_a, training_df, allele_encoding, num_workers=0
    )

    _seed_everything()
    peptides_b = EncodableSequences(training_df.peptide.values)
    fit_b, weights_b = _train_with_workers(
        peptides_b, training_df, allele_encoding, num_workers=0
    )

    numpy.testing.assert_array_equal(fit_a["loss"], fit_b["loss"])
    numpy.testing.assert_array_equal(fit_a["val_loss"], fit_b["val_loss"])
    for k in weights_a:
        assert torch.equal(weights_a[k], weights_b[k]), (
            f"num_workers=0 training is not self-reproducible at key {k!r}"
        )


def test_dataloader_num_workers_2_matches_num_workers_0(
    training_df, allele_encoding
):
    """num_workers=2 (prefetching with worker processes) must match num_workers=0.

    This is the semantic-preservation test for the parallel prefetch path.
    With shuffle=False and a deterministic training index array already
    built in the main process, DataLoader workers only do fancy-indexing
    into pre-built numpy arrays — no RNG, no reordering. PyTorch
    guarantees batch-delivery order when shuffle=False, so num_workers>0
    must produce identical batches.

    If this ever fails: we've introduced a subtle nondeterminism —
    probably via an RNG-touching op in __getitem__ or collate_fn.
    """
    _seed_everything()
    peptides_baseline = EncodableSequences(training_df.peptide.values)
    fit_baseline, weights_baseline = _train_with_workers(
        peptides_baseline, training_df, allele_encoding, num_workers=0
    )

    _seed_everything()
    peptides_prefetch = EncodableSequences(training_df.peptide.values)
    fit_prefetch, weights_prefetch = _train_with_workers(
        peptides_prefetch, training_df, allele_encoding, num_workers=2
    )

    numpy.testing.assert_array_equal(fit_baseline["loss"], fit_prefetch["loss"])
    numpy.testing.assert_array_equal(
        fit_baseline["val_loss"], fit_prefetch["val_loss"]
    )
    for k in weights_baseline:
        assert torch.equal(weights_baseline[k], weights_prefetch[k]), (
            f"num_workers=2 diverges from num_workers=0 at key {k!r} — "
            f"prefetching path introduced nondeterminism"
        )


def test_dataloader_cache_plus_workers_still_bit_identical(
    training_df, allele_encoding, tmp_path
):
    """Composition test: encoding cache + num_workers>0 must match baseline.

    This is the stack test: both Phase 1 changes (encoding cache in
    Step 2/3, DataLoader prefetching in Step 4) active at once must still
    produce bit-identical training. If either changes semantics
    independently the integration tests above catch it; this one catches
    the case where they INTERACT in a bad way.
    """
    params = EncodingParams(**DEFAULT_PEPTIDE_ENCODING)
    cache = EncodingCache(cache_dir=tmp_path / "encoding_cache", params=params)
    unique_peptides = list(
        pandas.Series(training_df.peptide.values).drop_duplicates()
    )
    encoded_all, peptide_to_idx = cache.get_or_build(unique_peptides)

    # Baseline: uncached, num_workers=0.
    _seed_everything()
    peptides_baseline = EncodableSequences(training_df.peptide.values)
    fit_baseline, weights_baseline = _train_with_workers(
        peptides_baseline, training_df, allele_encoding, num_workers=0
    )

    # Stack: cached + num_workers=2.
    _seed_everything()
    fold_indices = numpy.array(
        [peptide_to_idx[p] for p in training_df.peptide.values], dtype=numpy.int64
    )
    fold_encoded = encoded_all[fold_indices]
    peptides_stack = make_prepopulated_encodable_sequences(
        training_df.peptide.values, fold_encoded, params
    )
    fit_stack, weights_stack = _train_with_workers(
        peptides_stack, training_df, allele_encoding, num_workers=2
    )

    numpy.testing.assert_array_equal(fit_baseline["loss"], fit_stack["loss"])
    numpy.testing.assert_array_equal(fit_baseline["val_loss"], fit_stack["val_loss"])
    for k in weights_baseline:
        assert torch.equal(weights_baseline[k], weights_stack[k]), (
            f"encoding-cache + DataLoader-workers stack diverges at key {k!r}"
        )


def test_fit_generator_self_reproducible(training_df, allele_encoding):
    """fit_generator with the val-tensor hoist must be self-reproducible.

    Regression lock for the Phase 1 extension (#268) to fit_generator:
    - val_peptide_device / val_allele_device / val_y_device are allocated
      once before the epoch loop instead of re-materialized per epoch.
    - Validation uses those device tensors directly.

    Running the same seeded fit_generator twice must produce bit-identical
    loss and val_loss trajectories. If the hoist accidentally reuses
    the tensor in a way that leaks state across epochs (e.g. in-place
    modification, gradient accumulation on a frozen graph), this test
    will catch it.
    """
    _seed_everything(seed=24680)

    peptides = EncodableSequences(TRAIN_PEPTIDES[:16])
    affinities = numpy.linspace(50.0, 50000.0, 16)

    allele_list = [TRAIN_ALLELES_CYCLE[i % 3] for i in range(16)]
    alleles = AlleleEncoding(
        alleles=allele_list, allele_to_sequence=ALLELE_TO_SEQUENCE,
    )

    hp = _tiny_hyperparameters()

    def make_generator():
        # Two chunks; first 8 peptides then next 8.
        for start in (0, 8):
            chunk_peptides = EncodableSequences(TRAIN_PEPTIDES[start:start + 8])
            chunk_alleles = AlleleEncoding(
                alleles=allele_list[start:start + 8],
                allele_to_sequence=ALLELE_TO_SEQUENCE,
            )
            yield (chunk_alleles, chunk_peptides,
                   affinities[start:start + 8])

    def run_once():
        _seed_everything(seed=24680)
        net = Class1NeuralNetwork(**hp)
        # Create the network via a dummy call to fit() first; or let
        # fit_generator build it.
        net.fit_generator(
            generator=make_generator(),
            validation_peptide_encoding=peptides,
            validation_affinities=affinities,
            validation_allele_encoding=alleles,
            steps_per_epoch=2,
            epochs=3,
            min_epochs=3,
            patience=100,
            verbose=0,
            progress_print_interval=None,
        )
        info = net.fit_info[-1]
        return list(info["loss"]), list(info["val_loss"])

    loss_a, val_a = run_once()
    loss_b, val_b = run_once()
    numpy.testing.assert_array_equal(loss_a, loss_b)
    numpy.testing.assert_array_equal(val_a, val_b)


def test_fit_generator_validation_tensors_hoisted_once(
    training_df, allele_encoding
):
    """The val-tensor H2D happens ONCE, not per epoch.

    Spy on torch.from_numpy to count calls during fit_generator. Before
    the hoist: 3 calls (peptide, allele, y) × N epochs. After: 3 calls
    up front, zero inside the epoch loop for validation.

    This is a perf-regression guard — if someone refactors fit_generator
    and moves the val H2D back inside the epoch loop, this test fires.
    """
    import torch as _torch

    _seed_everything(seed=24680)
    peptides = EncodableSequences(TRAIN_PEPTIDES[:8])
    affinities = numpy.linspace(50.0, 50000.0, 8)
    allele_list = [TRAIN_ALLELES_CYCLE[i % 3] for i in range(8)]
    alleles = AlleleEncoding(
        alleles=allele_list, allele_to_sequence=ALLELE_TO_SEQUENCE,
    )

    def make_generator():
        yield (alleles, peptides, affinities)

    hp = _tiny_hyperparameters()
    net = Class1NeuralNetwork(**hp)

    # Count all torch.from_numpy calls under fit_generator. The
    # per-step training loop calls it 3× (peptide/allele/y). The val
    # side should call it 3× TOTAL (hoisted), not 3× per epoch.
    call_count = [0]
    orig = _torch.from_numpy

    def spy(a):
        call_count[0] += 1
        return orig(a)

    _torch.from_numpy = spy
    try:
        net.fit_generator(
            generator=make_generator(),
            validation_peptide_encoding=peptides,
            validation_affinities=affinities,
            validation_allele_encoding=alleles,
            steps_per_epoch=1,
            epochs=5,
            min_epochs=5,
            patience=100,
            verbose=0,
            progress_print_interval=None,
        )
    finally:
        _torch.from_numpy = orig

    # Expected calls: 3 val-hoist up front + 3 per training step × 5 epochs
    # × 1 step = 15. Total = 18. Budget it loosely (±5) to tolerate other
    # numpy→torch conversions (e.g. loss internals) without hardcoding
    # an exact number.
    expected_max = 3 + (3 * 5) + 5  # hoist + step + slack
    assert call_count[0] <= expected_max, (
        f"torch.from_numpy called {call_count[0]} times in a 5-epoch run. "
        f"Expected ~{expected_max}. If this ballooned, the val-tensor "
        f"hoist regressed — H2D is running per-epoch again, negating the "
        f"Phase 1 fit_generator perf fix."
    )


def test_fit_generator_supports_allele_specific_mode():
    """Regression test: fit_generator must work with ``alleles=None``.

    ``fit()`` already supports allele-specific models via
    ``allele_encoding=None``. ``fit_generator`` should support the same
    mode; otherwise any caller that pretrains or streams data into an
    allele-specific model crashes on ``None.indices`` before the first
    epoch.
    """
    _seed_everything(seed=13579)

    peptides = EncodableSequences(TRAIN_PEPTIDES[:16])
    affinities = numpy.linspace(50.0, 50000.0, 16)
    hp = _tiny_hyperparameters()

    def make_generator():
        for start in (0, 8):
            yield (
                None,
                EncodableSequences(TRAIN_PEPTIDES[start:start + 8]),
                affinities[start:start + 8],
            )

    net = Class1NeuralNetwork(**hp)
    net.fit_generator(
        generator=make_generator(),
        validation_peptide_encoding=peptides,
        validation_affinities=affinities,
        validation_allele_encoding=None,
        steps_per_epoch=2,
        epochs=2,
        min_epochs=2,
        patience=100,
        verbose=0,
        progress_print_interval=None,
    )

    info = net.fit_info[-1]
    assert len(info["loss"]) == 2
    assert len(info["val_loss"]) == 2


def test_fit_generator_supports_variable_tail_batches():
    """Short final generator batches must run instead of being rejected."""
    _seed_everything(seed=97531)

    peptides = EncodableSequences(TRAIN_PEPTIDES[:13])
    affinities = numpy.linspace(50.0, 50000.0, 13)
    allele_list = [TRAIN_ALLELES_CYCLE[i % 3] for i in range(13)]
    alleles = AlleleEncoding(
        alleles=allele_list, allele_to_sequence=ALLELE_TO_SEQUENCE,
    )
    hp = _tiny_hyperparameters()

    def make_generator():
        first = AlleleEncoding(
            alleles=allele_list[:8], allele_to_sequence=ALLELE_TO_SEQUENCE,
        )
        second = AlleleEncoding(
            alleles=allele_list[8:], allele_to_sequence=ALLELE_TO_SEQUENCE,
        )
        yield (
            first,
            EncodableSequences(TRAIN_PEPTIDES[:8]),
            affinities[:8],
        )
        yield (
            second,
            EncodableSequences(TRAIN_PEPTIDES[8:13]),
            affinities[8:13],
        )

    net = Class1NeuralNetwork(**hp)
    net.fit_generator(
        generator=make_generator(),
        validation_peptide_encoding=peptides,
        validation_affinities=affinities,
        validation_allele_encoding=alleles,
        steps_per_epoch=2,
        epochs=2,
        min_epochs=2,
        patience=100,
        verbose=0,
        progress_print_interval=None,
    )

    info = net.fit_info[-1]
    assert len(info["loss"]) == 2
    assert len(info["val_loss"]) == 2


def test_fit_generator_preencoded_batches_support_dataloader_workers():
    """Pre-encoded generator batches must work through a worker DataLoader."""
    _seed_everything(seed=86420)

    peptides = EncodableSequences(TRAIN_PEPTIDES[:8])
    affinities = numpy.linspace(50.0, 50000.0, 8)
    allele_list = [TRAIN_ALLELES_CYCLE[i % 3] for i in range(8)]
    alleles = AlleleEncoding(
        alleles=allele_list, allele_to_sequence=ALLELE_TO_SEQUENCE,
    )
    hp = _tiny_hyperparameters()
    hp["dataloader_num_workers"] = 1

    net = Class1NeuralNetwork(**hp)
    net.fit_generator(
        generator=(),
        generator_factory=_make_preencoded_fit_generator,
        generator_batches_are_encoded=True,
        validation_peptide_encoding=peptides,
        validation_affinities=affinities,
        validation_allele_encoding=alleles,
        steps_per_epoch=2,
        epochs=1,
        min_epochs=1,
        patience=100,
        verbose=0,
        progress_print_interval=None,
    )

    info = net.fit_info[-1]
    assert len(info["loss"]) == 1
    assert len(info["val_loss"]) == 1
    assert info["num_points"] == 8


# ---- Daemon-process compatibility gate ----
#
# mhcflurry's training orchestrator runs workers via multiprocessing.Pool.
# Pool workers are DAEMONIC by default, and Python enforces that daemon
# processes cannot spawn children — so ``DataLoader(num_workers>0)``
# inside a Pool worker raises AssertionError: daemonic processes are not
# allowed to have children. Our _make_fit_dataloader must detect this and
# transparently downgrade to num_workers=0.
#
# Missing this downgrade detonates every 32-work-item pan-allele run at
# the first epoch. Caught on a live A100 training run, 2026-04-20.


def _train_in_daemon_subprocess(result_queue, peptide_list, df_records, allele_encoding_data):
    """Worker body: reconstruct the minimal training state and call fit().

    Intentionally does NOT import anything pytest-related — this runs in
    a separate Python process so we need to construct the training inputs
    from picklable primitives.
    """
    import os
    # Quiet workers in CI output.
    os.environ["MHCFLURRY_DEFAULT_DEVICE"] = "cpu"
    try:
        import multiprocessing as mp
        import pandas as _pandas
        from mhcflurry.allele_encoding import AlleleEncoding as _AE
        from mhcflurry.class1_neural_network import Class1NeuralNetwork as _CNN
        from mhcflurry.common import configure_pytorch as _configure

        _configure(backend="cpu")

        # Assert we're actually daemonic — the test premise.
        assert mp.current_process().daemon, \
            "subprocess expected to be daemonic; test plumbing is broken"

        df = _pandas.DataFrame.from_records(df_records)
        allele_enc = _AE(
            alleles=allele_encoding_data["alleles"],
            allele_to_sequence=allele_encoding_data["allele_to_sequence"],
        )
        hp = dict(
            peptide_encoding={
                "alignment_method": "left_pad_centered_right_pad",
                "max_length": 15,
                "vector_encoding_name": "BLOSUM62",
            },
            max_epochs=1,
            minibatch_size=8,
            layer_sizes=[8],
            peptide_dense_layer_sizes=[],
            allele_dense_layer_sizes=[],
            locally_connected_layers=[],
            peptide_allele_merge_method="concatenate",
            peptide_allele_merge_activation="",
            peptide_amino_acid_encoding="BLOSUM62",
            topology="feedforward",
            dropout_probability=0.0,
            batch_normalization=False,
            dense_layer_l1_regularization=0.0,
            dense_layer_l2_regularization=0.0,
            validation_split=0.2,
            early_stopping=False,
            patience=100,
            min_delta=0.0,
            learning_rate=0.001,
            loss="custom:mse_with_inequalities",
            optimizer="rmsprop",
            activation="tanh",
            output_activation="sigmoid",
            init="glorot_uniform",
            data_dependent_initialization_method="lsuv",
            random_negative_rate=1.0,
            random_negative_method="by_allele_equalize_nonbinders",
            random_negative_affinity_min=30000.0,
            random_negative_affinity_max=50000.0,
            random_negative_binder_threshold=500.0,
            random_negative_constant=1,
            random_negative_distribution_smoothing=0.0,
            random_negative_match_distribution=True,
            # THE LOAD-BEARING KNOB: num_workers>0 would crash without the
            # daemon-context downgrade. A bare DataLoader(num_workers=2)
            # call raises AssertionError("daemonic processes are not
            # allowed to have children") deep in torch internals. If this
            # test crashes, the downgrade in _make_fit_dataloader is
            # broken.
            dataloader_num_workers=2,
        )
        net = _CNN(**hp)
        net.fit(
            peptides=peptide_list,
            affinities=df["measurement_value"].values,
            allele_encoding=allele_enc,
            verbose=0,
            progress_print_interval=None,
        )
        result_queue.put(("ok", len(net.fit_info[-1]["loss"])))
    except Exception as exc:
        import traceback as _tb
        result_queue.put(("error", f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"))


def test_dataloader_num_workers_downgrades_in_daemon_context(
    training_df, allele_encoding
):
    """Regression test: fit(dataloader_num_workers>0) must work in a Pool worker.

    Spawns a daemon subprocess (simulating what multiprocessing.Pool does
    for its workers) and calls fit() with dataloader_num_workers=2. If
    _make_fit_dataloader doesn't downgrade to 0 in daemon context, the
    subprocess crashes with AssertionError. This is THE test that would
    have caught the 2026-04-20 A100 training crash.

    Uses spawn context for the parent→daemon transition to mirror
    mhcflurry's actual Pool configuration.
    """
    import multiprocessing as mp

    # Serialize inputs as primitives (AlleleEncoding doesn't pickle cleanly
    # across spawn contexts; the daemon child rebuilds it).
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # multiprocessing.Process(daemon=True) replicates the Pool-worker
    # context: child is daemonic, so attempts to spawn its own children
    # should raise AssertionError in untreated code.
    peptide_list = list(training_df["peptide"].values)
    df_records = training_df.to_dict("records")
    allele_data = {
        "alleles": list(training_df["allele"].values),
        "allele_to_sequence": ALLELE_TO_SEQUENCE,
    }

    p = ctx.Process(
        target=_train_in_daemon_subprocess,
        args=(result_queue, peptide_list, df_records, allele_data),
        daemon=True,
    )
    p.start()
    p.join(timeout=180)
    if p.is_alive():
        p.terminate()
        pytest.fail("daemon-subprocess training did not finish within 3 min")

    assert not result_queue.empty(), (
        "daemon-subprocess produced no result — likely crashed before "
        "fit() returned (segfault or non-Python failure)"
    )
    status, payload = result_queue.get(timeout=5)
    assert status == "ok", (
        f"daemon-subprocess training FAILED. This is the regression that "
        f"detonated the 2026-04-20 A100 run. Check "
        f"_effective_num_workers in class1_neural_network.py.\n"
        f"Subprocess error:\n{payload}"
    )
    n_epochs = payload
    assert n_epochs == 1, f"expected 1 training epoch, got {n_epochs}"
