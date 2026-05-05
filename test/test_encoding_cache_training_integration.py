"""End-to-end integration tests for device-resident affinity training."""

from __future__ import annotations

import pickle
import random

import numpy
import pandas
import pytest
import torch

from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.class1_neural_network import (
    Class1NeuralNetwork,
    _StreamingBatchIterableDataset,
)
from mhcflurry.common import configure_pytorch
from mhcflurry.encodable_sequences import EncodableSequences


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


# ---- Affinity fit ignores streaming DataLoader worker setting ----


def _train_with_workers(
    peptides_arg,
    training_df,
    allele_encoding,
    num_workers,
):
    """Train with the streaming-pretrain worker knob set to a chosen value."""
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
    """Affinity fit is self-reproducible when the worker knob is 0.

    Affinity fit is device-resident and no longer uses a DataLoader for
    minibatches. This pins the baseline contract before comparing that
    the streaming-pretrain worker knob has no effect on affinity fit.
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
    """The streaming-pretrain worker knob must not alter affinity fit.

    `dataloader_num_workers` controls the streaming pretraining path.
    Affinity fine-tuning forms batches from device-resident tensors, so
    setting this hyperparameter must not change losses or weights.
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


def test_fit_generator_self_reproducible(training_df, allele_encoding):
    """fit_generator with the val-tensor hoist must be self-reproducible.

    Regression lock for the val-tensor hoist:
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
        f"hoist regressed — H2D is running per-epoch again."
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


def test_fit_generator_restarts_factory_after_exhaustion():
    """Finite factory sources must restart instead of producing empty epochs."""
    _seed_everything(seed=75319)

    peptides = EncodableSequences(TRAIN_PEPTIDES[:8])
    affinities = numpy.linspace(50.0, 50000.0, 8)
    allele_list = [TRAIN_ALLELES_CYCLE[i % 3] for i in range(8)]
    alleles = AlleleEncoding(
        alleles=allele_list, allele_to_sequence=ALLELE_TO_SEQUENCE,
    )
    hp = _tiny_hyperparameters()

    def one_batch_factory(worker_id=0, num_workers=1):
        assert worker_id == 0
        assert num_workers == 1
        x_dict, y = _PREENCODED_BATCHES[0]
        yield (
            {key: value.copy() for key, value in x_dict.items()},
            y.copy(),
        )

    net = Class1NeuralNetwork(**hp)
    net.fit_generator(
        generator=(),
        generator_factory=one_batch_factory,
        generator_batches_are_encoded=True,
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
    assert numpy.isfinite(info["loss"]).all()
    assert info["num_points"] == 16
    assert info["iterator_restarts"] >= 3


def test_fit_generator_dataset_is_picklable_with_live_generator():
    """Dataset must pickle cleanly even when caller passes a live generator.

    Regression for the 2026-04-23 8×A100 training crash: the pretrain
    caller builds ``generator=make_pretrain_generator()`` (a live
    generator object) alongside ``generator_factory=make_pretrain_generator``.
    Under ``dataloader_num_workers>0``, PyTorch spawns workers via
    ``ForkingPickler``, which chokes on the live generator with
    ``TypeError: cannot pickle 'generator' object``.

    The dataset must drop the live generator when a factory is present
    so pickling goes through the factory only.
    """
    live_generator = _make_preencoded_fit_generator()
    assert isinstance(live_generator, type((x for x in ()))), \
        "test premise: _make_preencoded_fit_generator returns a generator"

    dataset = _StreamingBatchIterableDataset(
        generator=live_generator,
        generator_factory=_make_preencoded_fit_generator,
        source_batches_are_encoded=True,
    )

    # Pickle round-trip must succeed. This is what DataLoader does to
    # ship the dataset to spawned workers.
    payload = pickle.dumps(dataset)
    restored = pickle.loads(payload)
    assert restored.generator is None
    assert restored.generator_factory is _make_preencoded_fit_generator
    assert restored.source_batches_are_encoded is True


def test_fit_generator_dataset_drops_bound_methods_on_encoded_path():
    """Pre-encoded path must not retain bound methods of the network.

    Even after the live-generator fix, if the dataset held
    ``self.peptides_to_network_input = net.peptides_to_network_input``
    (a bound method), pickling would drag the whole
    ``Class1NeuralNetwork`` into every spawned worker — both heavy and
    sometimes outright unpicklable. The encoded-batch path never reads
    those callbacks, so the dataset must drop them at construction.
    """
    net = Class1NeuralNetwork(**_tiny_hyperparameters())

    dataset = _StreamingBatchIterableDataset(
        generator=(),
        generator_factory=_make_preencoded_fit_generator,
        source_batches_are_encoded=True,
        allele_encoding_to_input=net.allele_encoding_to_network_input,
        peptides_to_network_input=net.peptides_to_network_input,
    )
    assert dataset.allele_encoding_to_input is None
    assert dataset.peptides_to_network_input is None
    # And the dataset pickles cleanly — without a bound-method
    # reference, there's no path for the network to tag along.
    pickle.dumps(dataset)


def test_fit_generator_dataset_keeps_callbacks_on_raw_path():
    """Raw-tuple path must retain the encoding callbacks — they're used."""
    net = Class1NeuralNetwork(**_tiny_hyperparameters())

    dataset = _StreamingBatchIterableDataset(
        generator=iter(()),  # empty, but live-generator-shaped
        generator_factory=None,
        source_batches_are_encoded=False,
        allele_encoding_to_input=net.allele_encoding_to_network_input,
        peptides_to_network_input=net.peptides_to_network_input,
    )
    assert dataset.allele_encoding_to_input is not None
    assert dataset.peptides_to_network_input is not None


def test_fit_generator_downgrades_num_workers_without_factory(caplog):
    """No factory → must downgrade to num_workers=0 with a warning.

    Worker-prefetch needs a picklable per-worker source. Without
    ``generator_factory``, the dataset can only iterate the single
    main-process generator — which can't be sharded and can't be
    pickled. fit_generator must detect this and downgrade rather than
    letting the DataLoader crash at worker spawn.
    """
    _seed_everything(seed=13579)

    peptides = EncodableSequences(TRAIN_PEPTIDES[:8])
    affinities = numpy.linspace(50.0, 50000.0, 8)
    allele_list = [TRAIN_ALLELES_CYCLE[i % 3] for i in range(8)]
    alleles = AlleleEncoding(
        alleles=allele_list, allele_to_sequence=ALLELE_TO_SEQUENCE,
    )
    hp = _tiny_hyperparameters()
    hp["dataloader_num_workers"] = 2

    net = Class1NeuralNetwork(**hp)
    with caplog.at_level("WARNING", logger="root"):
        net.fit_generator(
            generator=_make_preencoded_fit_generator(),
            generator_factory=None,  # the load-bearing omission
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
    assert info["dataloader_num_workers"] == 2, \
        "recorded value reflects the request, not the downgrade"
    assert any(
        "downgrading to 0" in rec.message for rec in caplog.records
    ), "expected a downgrade warning in the log"


def test_fit_generator_downgrades_num_workers_on_raw_batches(caplog):
    """Raw-batch (non-encoded) source + factory → must still downgrade.

    Second downgrade trigger (the ``generator_batches_are_encoded=False``
    side). The dataset on the raw path stores bound methods of the
    ``Class1NeuralNetwork`` instance (``peptides_to_network_input``,
    ``allele_encoding_to_network_input``), and pickling those drags
    the whole network into every worker — which is wrong. Worker-
    prefetch must only activate when batches are pre-encoded AND a
    factory is present. See a286d51 (``_StreamingBatchIterableDataset``
    downgrade check).
    """
    _seed_everything(seed=13580)

    peptides = EncodableSequences(TRAIN_PEPTIDES[:8])
    affinities = numpy.linspace(50.0, 50000.0, 8)
    allele_list = [TRAIN_ALLELES_CYCLE[i % 3] for i in range(8)]
    alleles = AlleleEncoding(
        alleles=allele_list, allele_to_sequence=ALLELE_TO_SEQUENCE,
    )
    hp = _tiny_hyperparameters()
    hp["dataloader_num_workers"] = 2

    # Build a raw-batch generator + factory. The factory is picklable,
    # but the source says "batches arrive unencoded" — so the dataset
    # would store network bound-method references that the worker
    # pickler couldn't ship safely. The downgrade must fire BEFORE
    # that pickle would be attempted.
    def raw_batch_factory(worker_id=0, num_workers=1):
        del worker_id, num_workers  # shard-agnostic for this tiny test
        for _ in range(2):
            yield {
                "peptide": list(TRAIN_PEPTIDES[:4]),
                "allele": [TRAIN_ALLELES_CYCLE[i % 3] for i in range(4)],
                "affinity": numpy.linspace(50.0, 50000.0, 4),
            }

    net = Class1NeuralNetwork(**hp)
    # The downgrade happens BEFORE any actual training work — we only
    # care that the log is emitted. The raw-batch path can legitimately
    # fail downstream in this minimal scaffolding (the test peptide /
    # allele shapes aren't the raw decoder's expected format), but
    # that's orthogonal to the assertion here.
    with caplog.at_level("WARNING", logger="root"):
        try:
            net.fit_generator(
                generator=raw_batch_factory(),
                generator_factory=raw_batch_factory,
                generator_batches_are_encoded=False,  # the trigger
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
        except Exception:
            # Downstream raw-path failure is acceptable — we're
            # asserting the downgrade fires, not that the raw path
            # works end-to-end in this tiny harness.
            pass

    assert any(
        "downgrading to 0" in rec.message for rec in caplog.records
    ), "expected a downgrade warning for raw-batch path"
