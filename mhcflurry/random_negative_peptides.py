import json
import logging
import math
import os

import numpy
import pandas

from . import amino_acid
from .encodable_sequences import EncodableSequences
from .hyperparameters import HyperparameterDefaults
from .common import amino_acid_distribution, random_peptides


class RandomNegativePeptides(object):
    """
    Generate random negative (peptide, allele) pairs. These are used during
    model training, where they are resampled at each epoch.
    """

    hyperparameter_defaults = HyperparameterDefaults(
        random_negative_rate=0.0,
        random_negative_constant=0,
        random_negative_match_distribution=True,
        random_negative_distribution_smoothing=0.0,
        random_negative_method="recommended",
        random_negative_binder_threshold=None,
        random_negative_lengths=[8,9,10,11,12,13,14,15])
    """
    Hyperperameters for random negative peptides.

    Number of random negatives will be:
        random_negative_rate * (num measurements) + random_negative_constant

    where the exact meaning of (num measurements) depends on the particular
    random_negative_method in use.

    If random_negative_match_distribution is True, then the amino acid
    frequencies of the training data peptides are used to generate the
    random peptides.

    Valid values for random_negative_method are:
        "by_length": used for allele-specific prediction. See description in
            `RandomNegativePeptides.plan_by_length` method.
        "by_allele": used for pan-allele prediction. See
            `RandomNegativePeptides.plan_by_allele` method.
        "by_allele_equalize_nonbinders": used for pan-allele prediction. See
            `RandomNegativePeptides.plan_by_allele_equalize_nonbinders` method.
        "recommended": the default. Use by_length if the predictor is allele-
            specific and by_allele if it's pan-allele.

    """

    def __init__(self, **hyperparameters):
        self.hyperparameters = self.hyperparameter_defaults.with_defaults(
            hyperparameters)
        self.plan_df = None
        self.aa_distribution = None

    def plan(self, peptides, affinities, alleles=None, inequalities=None):
        """
        Calculate the number of random negatives for each allele and peptide
        length. Call this once after instantiating the object.

        Parameters
        ----------
        peptides : list of string
        affinities : list of float
        alleles : list of string, optional
        inequalities : list of string (">", "<", or "="), optional

        Returns
        -------
        pandas.DataFrame indicating number of random negatives for each length
        and allele.
        """
        numpy.testing.assert_equal(len(peptides), len(affinities))
        if alleles is not None:
            numpy.testing.assert_equal(len(peptides), len(alleles))
        if inequalities is not None:
            numpy.testing.assert_equal(len(peptides), len(inequalities))

        peptides = pandas.Series(peptides, copy=False)
        peptide_lengths = peptides.str.len()

        if self.hyperparameters['random_negative_match_distribution']:
            self.aa_distribution = amino_acid_distribution(
                peptides.values,
                smoothing=self.hyperparameters[
                    'random_negative_distribution_smoothing'
                ])
            logging.info(
                "Using amino acid distribution for random negative:\n%s" % (
                    str(self.aa_distribution.to_dict())))

        df_all = pandas.DataFrame({
            'length': peptide_lengths,
            'affinity': affinities,
        })
        df_all["allele"] = "" if alleles is None else alleles
        df_all["inequality"] = "=" if inequalities is None else inequalities

        df_binders = None
        df_nonbinders = None
        if self.hyperparameters['random_negative_binder_threshold']:
            df_nonbinders = df_all.loc[
                (df_all.inequality != "<") &
                (df_all.affinity > self.hyperparameters[
                    'random_negative_binder_threshold'
                ])
            ]
            df_binders = df_all.loc[
                (df_all.inequality != ">") &
                (df_all.affinity <= self.hyperparameters[
                    'random_negative_binder_threshold'
                ])
            ]

        method = self.hyperparameters['random_negative_method']
        if method == 'recommended':
            # by_length for allele-specific prediction and by_allele for pan.
            method = (
                "by_length"
                if alleles is None else
                "by_allele")

        function = {
            'by_length': self.plan_by_length,
            'by_allele': self.plan_by_allele,
            'by_allele_equalize_nonbinders':
                self.plan_by_allele_equalize_nonbinders,
        }[method]
        function(df_all, df_binders, df_nonbinders)
        assert self.plan_df is not None
        logging.info("Random negative plan [%s]:\n%s", method, self.plan_df)
        return self.plan_df

    def plan_by_length(self, df_all, df_binders=None, df_nonbinders=None):
        """
        Generate a random negative plan using the "by_length" policy.

        Parameters are as in the `plan` method. No return value.

        Used for allele-specific predictors. Does not work well for pan-allele.

        Different numbers of random negatives per length. Alleles are sampled
        proportionally to the number of times they are used in the training
        data.
        """
        assert list(df_all.allele.unique()) == [""], (
            "by_length only recommended for allele specific prediction")

        df = df_all if df_binders is None else df_binders
        lengths = self.hyperparameters['random_negative_lengths']

        length_to_num_random_negative = {}
        length_counts = df.length.value_counts().to_dict()
        for length in lengths:
            length_to_num_random_negative[length] = int(
                length_counts.get(length, 0) *
                self.hyperparameters['random_negative_rate'] +
                self.hyperparameters['random_negative_constant'])

        plan_df = pandas.DataFrame(index=sorted(df.allele.unique()))
        for length in lengths:
            plan_df[length] = length_to_num_random_negative[length]
        self.plan_df = plan_df.astype(int)

    def plan_by_allele(self, df_all, df_binders=None, df_nonbinders=None):
        """
        Generate a random negative plan using the "by_allele" policy.

        Parameters are as in the `plan` method. No return value.

        For each allele, a particular number of random negatives are used
        for all lengths. Across alleles, the number of random negatives
        varies; within an allele, the number of random negatives for each
        length is a constant
        """
        allele_to_num_per_length = {}
        total_random_peptides_per_length = 0
        df = df_all if df_binders is None else df_binders
        lengths = self.hyperparameters['random_negative_lengths']
        all_alleles = df_all.allele.unique()
        for allele in all_alleles:
            sub_df = df.loc[df.allele == allele]
            num_for_allele = len(sub_df) * (
                self.hyperparameters['random_negative_rate']
            ) + self.hyperparameters['random_negative_constant']
            num_per_length = int(math.ceil(
                num_for_allele / len(lengths)))
            total_random_peptides_per_length += num_per_length
            allele_to_num_per_length[allele] = num_per_length

        plan_df = pandas.DataFrame(index=sorted(df.allele.unique()))
        for length in lengths:
            plan_df[length] = plan_df.index.map(allele_to_num_per_length)
        self.plan_df = plan_df.astype(int)

    def plan_by_allele_equalize_nonbinders(
            self, df_all, df_binders, df_nonbinders):
        """
        Generate a random negative plan using the
        "by_allele_equalize_nonbinders" policy.

        Parameters are as in the `plan` method. No return value.

        Requires that the random_negative_binder_threshold hyperparameter is set.

        In a first step, the number of random negatives selected by the
        "by_allele" method are added (see `plan_by_allele`). Then, the total
        number of non-binders are calculated for each allele and length. This
        total includes non-binder measurements in the training data plus the
        random negative peptides added in the first step. In a second step,
        additional random negative peptides are added so that for each allele,
        all peptide lengths have the same total number of non-binders.
        """
        assert df_binders is not None
        assert df_nonbinders is not None

        lengths = self.hyperparameters['random_negative_lengths']

        self.plan_by_allele(df_all, df_binders, df_nonbinders)
        first_pass_plan = self.plan_df
        self.plan_df = None

        # Use floating point while populating so NaN assignment remains valid
        # across pandas versions; cast to int at the end.
        new_plan = first_pass_plan.astype(float).copy()
        new_plan[:] = numpy.nan

        for (allele, first_pass_per_length) in first_pass_plan.iterrows():
            real_nonbinders_by_length = df_nonbinders.loc[
                df_nonbinders.allele == allele
            ].length.value_counts().reindex(lengths).fillna(0)
            total_nonbinders_by_length = (
                real_nonbinders_by_length + first_pass_per_length)
            new_plan.loc[allele] = first_pass_per_length + (
                total_nonbinders_by_length.max() - total_nonbinders_by_length)

        if new_plan.isna().any().any():
            raise AssertionError(
                "Random negative plan contains NaN after equalization; "
                "this indicates an incomplete per-allele assignment bug."
            )

        self.plan_df = new_plan.astype(int)

    def get_alleles(self):
        """
        Get the list of alleles corresponding to each random negative peptide
        as returned by `get_peptides`. This does NOT change and can be safely
        called once and reused.

        Returns
        -------
        list of string
        """
        assert self.plan_df is not None, "Call plan() first"
        alleles = []
        for allele, row in self.plan_df.iterrows():
            alleles.extend([allele] * int(row.sum()))
        assert len(alleles) == self.get_total_count()
        return alleles

    def get_peptides(self, rng=None):
        """
        Get the list of random negative peptides. This will be different each
        time the method is called.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            When supplied, all random draws go through this generator.
            Lets a caller make the returned peptide list deterministic —
            used by ``RandomNegativesPool`` to build reproducible
            multi-epoch pools. When None the numpy global state is used,
            preserving the historical "fresh peptides every call"
            behavior.

        Returns
        -------
        list of string

        """
        assert self.plan_df is not None, "Call plan() first"
        peptides = []
        for allele, row in self.plan_df.iterrows():
            for (length, num) in row.items():
                peptides.extend(
                    random_peptides(
                        num,
                        length=length,
                        distribution=self.aa_distribution,
                        rng=rng))
        assert len(peptides) == self.get_total_count()
        return peptides

    def get_total_count(self):
        """
        Total number of planned random negative peptides.

        Returns
        -------
        int
        """
        return self.plan_df.sum().sum()


class RandomNegativesPool(object):
    """Amortize random-negative generation and encoding across N epochs.

    Pre-issue-#268 Phase 1, ``Class1NeuralNetwork.fit()`` called
    ``planner.get_peptides()`` followed by ``peptides_to_network_input``
    at the top of every epoch — profiling on the release-exact 8xA100
    run showed that pair at ~17 s/epoch (~44% of epoch wall-clock) for
    the pan-allele default random-negative counts. The peptide strings
    themselves are tiny; the cost is almost entirely in the
    BLOSUM62-encoding pass.

    This class generates ``pool_epochs`` worth of random peptides in one
    call to the planner, encodes the whole pool once, and hands back an
    O(1) slice per epoch. Setting ``pool_epochs=1`` reproduces the
    pre-pool semantics exactly (one generation + one encode per epoch).
    Setting ``pool_epochs=100`` reduces the amortized per-epoch encode
    cost by ~100x; the trade-off is that within a pool-cycle the
    negatives no longer *change* every epoch — consecutive epochs in a
    cycle see distinct slices of the same pool, not freshly-sampled
    peptides. A new pool is generated at the start of each cycle
    (epoch // pool_epochs boundary).

    Seeding is optional. When ``seed`` is None the peptides are drawn
    from the process's numpy global state, matching the pre-pool
    semantics: workers in a training pool diverge naturally because
    they are separate processes with independent RNG state. Supplying
    an explicit seed makes pool contents reproducible — useful for
    debugging and for regression tests.
    """

    def __init__(
            self,
            planner,
            peptide_encoder,
            pool_epochs=1,
            seed=None,
            device=None,
            peptide_encoding=None):
        """
        Parameters
        ----------
        planner : RandomNegativePeptides
            A planner that has already had ``plan(...)`` called on it.

        peptide_encoder : callable
            Receives an ``EncodableSequences`` and returns an encoded
            numpy array (typically ``Class1NeuralNetwork
            .peptides_to_network_input``). Called once per cycle on the
            full pool. Ignored when ``device`` is set — the device path
            samples and encodes integer indices directly.

        pool_epochs : int
            Number of consecutive epochs that share a pool. 1 is
            semantically identical to pre-Phase-1 behavior.

        seed : int, optional
            Seed for the per-cycle RNG. When None, draws go through
            numpy's global state (or, on the device path, an unseeded
            ``torch.Generator`` on ``device``).

        device : torch.device or str, optional
            When set, the pool builds device-resident int8 tensors via
            :func:`encode_random_negatives_on_device` instead of the
            host-side string-→encoder path. ``peptide_encoding`` must
            be provided alongside; ``peptide_encoder`` is unused on
            this path. Peptide strings are not materialized
            (``get_epoch_inputs`` returns ``(None, encoded_slice)``).

        peptide_encoding : dict, optional
            Required when ``device`` is set. Subset of the model's
            ``peptide_encoding`` hyperparameter — at minimum
            ``alignment_method`` and ``max_length`` (plus
            ``left_edge`` / ``right_edge`` for ``pad_middle``).
        """
        assert planner.plan_df is not None, "Call planner.plan() first"
        if device is not None and peptide_encoding is None:
            raise ValueError(
                "RandomNegativesPool: device-resident pool requires "
                "peptide_encoding (alignment_method, max_length, ...)"
            )
        self.planner = planner
        self.peptide_encoder = peptide_encoder
        self.pool_epochs = max(int(pool_epochs), 1)
        self.seed = seed
        self.device = device
        self.peptide_encoding = peptide_encoding
        self._total_count = int(planner.get_total_count())
        self._current_cycle = None
        self._current_encoded = None  # shape: (pool_epochs * total_count, ...)
        self._current_peptides = None  # list of str, len = pool_epochs * total_count
        # Allocated lazily on first device-mode build, reused per cycle.
        self._device_buffer = None

    def _rng_for_cycle(self, cycle):
        if self.seed is None:
            return None
        seed_seq = numpy.random.SeedSequence(
            entropy=int(self.seed), spawn_key=(int(cycle),)
        )
        return numpy.random.default_rng(seed_seq)

    def _torch_generator_for_cycle(self, cycle):
        """Per-cycle torch.Generator for the device path.

        Returns ``None`` when ``self.seed`` is None (let torch's global
        state vary per worker). Mirrors the SeedSequence(seed, cycle)
        deterministic mix used by the host path so seeded device pools
        are reproducible across rebuilds.
        """
        import torch as _torch
        if self.seed is None:
            return None
        seed_seq = numpy.random.SeedSequence(
            entropy=int(self.seed), spawn_key=(int(cycle),)
        )
        # Squeeze the SeedSequence into a 64-bit unsigned int seed.
        seed64 = int(seed_seq.generate_state(2, dtype=numpy.uint32).astype(
            numpy.uint64
        )[0]) | (
            int(seed_seq.generate_state(2, dtype=numpy.uint32).astype(
                numpy.uint64
            )[1]) << 32
        )
        seed64 &= (1 << 63) - 1  # torch wants signed positive 64-bit
        gen = _torch.Generator(device=self.device)
        gen.manual_seed(seed64)
        return gen

    def _build_cycle(self, cycle):
        if self.device is not None:
            self._build_cycle_device(cycle)
            return
        rng = self._rng_for_cycle(cycle)
        peptides = []
        for _ in range(self.pool_epochs):
            peptides.extend(self.planner.get_peptides(rng=rng))
        assert len(peptides) == self.pool_epochs * self._total_count
        encoded = self.peptide_encoder(EncodableSequences.create(peptides))
        self._current_peptides = peptides
        self._current_encoded = encoded
        self._current_cycle = cycle

    def _build_cycle_device(self, cycle):
        """Device-resident path: sample + encode int indices on device.

        Allocates a single device-resident int8 tensor on first call
        and refills it in place every cycle. ``_current_peptides``
        stays ``None`` — fit() doesn't need the strings; logging /
        debug callers can re-materialize via
        :func:`amino_acid.indices_to_peptide`.
        """
        gen = self._torch_generator_for_cycle(cycle)
        if self._device_buffer is None:
            self._device_buffer = encode_random_negatives_on_device(
                planner=self.planner,
                pool_epochs=self.pool_epochs,
                peptide_encoding=self.peptide_encoding,
                device=self.device,
                generator=gen,
            )
        else:
            encode_random_negatives_on_device(
                planner=self.planner,
                pool_epochs=self.pool_epochs,
                peptide_encoding=self.peptide_encoding,
                device=self.device,
                generator=gen,
                out=self._device_buffer,
            )
        self._current_encoded = self._device_buffer
        self._current_peptides = None
        self._current_cycle = cycle

    def get_epoch_inputs(self, epoch):
        """Return ``(peptides_list, encoded_slice)`` for ``epoch``.

        The returned ``encoded_slice`` is a view into the pool-level
        encoded array — no copy — so downstream assignments that expect
        to own the memory should ``.copy()`` it first. The list of raw
        peptide strings is included for callers that need it (e.g.
        logging / diagnostics).

        On the device-resident path ``peptides_list`` is ``None`` (the
        strings are never materialized; recover them via
        :func:`amino_acid.indices_to_peptide` against the int8
        encoded slice if a logging caller needs them).

        Phase 3 (#268) shared-mmap path: when the pool was built via
        ``from_shared_mmap`` with a ``permutation_seed``, the slice is
        reordered by a per-worker permutation seeded by that value mixed
        with the epoch counter. Diversity is preserved even though all
        workers read from the same byte-identical encoded array.
        """
        max_epoch = getattr(self, "_mmap_max_epoch", None)
        if max_epoch is not None and int(epoch) > max_epoch:
            raise ValueError(
                "Shared-mmap RandomNegativesPool was sized for "
                "pool_epochs=%d (epochs 0-%d); caller requested "
                "epoch=%d. Size the pool to max_epochs before "
                "launching workers, or switch to an in-process "
                "RandomNegativesPool that can rebuild per cycle." % (
                    self.pool_epochs, max_epoch, int(epoch),
                )
            )
        cycle = int(epoch) // self.pool_epochs
        if cycle != self._current_cycle:
            self._build_cycle(cycle)
        offset = (int(epoch) % self.pool_epochs) * self._total_count
        end = offset + self._total_count
        if self._current_peptides is None:
            peptides_slice = None
        else:
            peptides_slice = self._current_peptides[offset:end]
        encoded_slice = self._current_encoded[offset:end]
        permutation = self._epoch_permutation(epoch)
        if permutation is not None:
            if peptides_slice is not None:
                peptides_slice = [peptides_slice[i] for i in permutation]
            if self.device is not None:
                import torch as _torch
                perm_t = _torch.as_tensor(
                    permutation, device=encoded_slice.device,
                    dtype=_torch.long,
                )
                encoded_slice = encoded_slice.index_select(0, perm_t)
            else:
                encoded_slice = numpy.asarray(encoded_slice)[permutation]
        return peptides_slice, encoded_slice

    @property
    def total_count(self):
        return self._total_count

    # --- Phase 3 (#268): shared-mmap pool primitive ---
    #
    # In a multi-worker training pool (the pan-allele release_exact run
    # spins up 16 training workers), each worker currently holds its own
    # copy of the pool-epoch encoded array: ~17 MB × 16 = ~272 MB of RSS
    # duplicated across processes. Workers shouldn't need distinct
    # peptides — cross-worker diversity comes from the per-worker
    # permutation over the pool, not from the pool contents themselves
    # (see the ``random_negative_seed`` threaded from work_item identity
    # in train_pan_allele_models_command).
    #
    # The API below lets a coordinator (typically the training driver
    # before it forks workers) encode a deterministic pool and persist it as
    # an int8 memmap + JSON manifest. The writer streams one epoch-slice at
    # a time so the coordinator never has to hold the full pool in RAM.
    # Workers then load the pool with
    # ``from_shared_mmap`` — the OS page cache backs all of them with
    # a single resident copy, cutting RSS ~N× across a pool of size N.
    # Within a worker, ``get_epoch_inputs`` continues to return an
    # ordinary numpy view into the encoded array; optional
    # ``permutation_seed`` shuffles that view deterministically per
    # worker so diversity is preserved.
    #
    # Not yet wired into ``local_parallelism`` / ``fit()`` — this
    # change just delivers the IPC primitive. Wiring it up needs a
    # coordinator hook inside the NonDaemonPool that runs once per
    # worker spawn; that's a follow-up PR.

    _MANIFEST_NAME = "random_negatives_pool.json"
    _ENCODED_NAME = "random_negatives_encoded.int8.mmap"
    _PEPTIDES_NAME = "random_negatives_peptides.json"

    @classmethod
    def write_shared_pool(
            cls,
            output_dir,
            planner,
            peptide_encoder,
            pool_epochs,
            seed):
        """Generate a deterministic pool and persist it under ``output_dir``.

        Creates three files:
          - ``random_negatives_encoded.int8.mmap`` — the (pool_epochs *
            total_count, ...) encoded array, stored int8 (values are in
            [-128, 127]; BLOSUM62 fits tightly in that range, index
            payloads trivially so). Callers that need a non-int8
            encoding can skip ``write_shared_pool`` and use the
            in-process ``RandomNegativesPool`` directly.
          - ``random_negatives_peptides.json`` — the raw peptide strings
            in the same row order; kept for debugging/logging.
          - ``random_negatives_pool.json`` — manifest (shape, dtype,
            pool_epochs, total_count, seed, encoded_file, peptides_file).

        Workers load the pool via ``from_shared_mmap`` and read the
        encoded array as mmap — one page-cache copy shared across the
        worker pool instead of N per-process copies.
        """
        if seed is None:
            raise ValueError(
                "write_shared_pool requires seed to be set — the whole "
                "point is deterministic content that workers share."
            )
        builder = cls(
            planner=planner,
            peptide_encoder=peptide_encoder,
            pool_epochs=pool_epochs,
            seed=seed,
        )
        pool_epochs = builder.pool_epochs
        total_count = builder._total_count

        os.makedirs(output_dir, exist_ok=True)
        encoded_path = os.path.join(output_dir, cls._ENCODED_NAME)
        encoded_tmp_path = f"{encoded_path}.tmp.{os.getpid()}"
        peptides_path = os.path.join(output_dir, cls._PEPTIDES_NAME)
        peptides_tmp_path = f"{peptides_path}.tmp.{os.getpid()}"
        manifest_path = os.path.join(output_dir, cls._MANIFEST_NAME)
        manifest_tmp_path = f"{manifest_path}.tmp.{os.getpid()}"

        rng = builder._rng_for_cycle(0)
        mm = None
        shape = None
        tmp_paths = [encoded_tmp_path, peptides_tmp_path, manifest_tmp_path]
        try:
            with open(peptides_tmp_path, "w", encoding="utf-8") as peptides_fd:
                peptides_fd.write("[")
                first_peptide = True
                for epoch_offset in range(pool_epochs):
                    peptides = planner.get_peptides(rng=rng)
                    assert len(peptides) == total_count
                    encoded = numpy.asarray(
                        peptide_encoder(EncodableSequences.create(peptides))
                    )
                    if len(encoded) != total_count:
                        raise ValueError(
                            "Random negative encoder returned %d rows for %d "
                            "peptides" % (len(encoded), total_count)
                        )
                    encoded_int8 = encoded.astype("int8", copy=False)
                    if mm is None:
                        shape = (
                            pool_epochs * total_count,
                            *encoded_int8.shape[1:],
                        )
                        mm = numpy.memmap(
                            encoded_tmp_path,
                            dtype="int8",
                            mode="w+",
                            shape=shape,
                        )
                    elif encoded_int8.shape[1:] != shape[1:]:
                        raise ValueError(
                            "Random negative encoder shape changed from %r "
                            "to %r" % (shape[1:], encoded_int8.shape[1:])
                        )
                    start = epoch_offset * total_count
                    mm[start : start + total_count] = encoded_int8
                    for peptide in peptides:
                        if not first_peptide:
                            peptides_fd.write(",")
                        json.dump(peptide, peptides_fd)
                        first_peptide = False
                    del encoded, encoded_int8, peptides
                peptides_fd.write("]")
            if mm is not None:
                mm.flush()
                del mm
                mm = None

            manifest = {
                "shape": list(shape),
                "dtype": "int8",
                "pool_epochs": int(pool_epochs),
                "total_count": int(total_count),
                "seed": int(seed),
                "encoded_file": cls._ENCODED_NAME,
                "peptides_file": cls._PEPTIDES_NAME,
            }
            with open(manifest_tmp_path, "w", encoding="utf-8") as fd:
                json.dump(manifest, fd, indent=2, sort_keys=True)
            os.replace(encoded_tmp_path, encoded_path)
            os.replace(peptides_tmp_path, peptides_path)
            os.replace(manifest_tmp_path, manifest_path)
            tmp_paths = []
        finally:
            if mm is not None:
                del mm
            for path in tmp_paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        return manifest_path

    @classmethod
    def from_shared_mmap(
            cls,
            output_dir,
            planner,
            peptide_encoder=None,
            permutation_seed=None):
        """Load a pool written by ``write_shared_pool`` as a shared mmap.

        Parameters
        ----------
        output_dir : str
            Directory containing the files written by ``write_shared_pool``.
        planner : RandomNegativePeptides
            The worker's planner. Must match the planner used to write
            the pool (same ``get_total_count()``); otherwise the slice
            bounds misalign.
        peptide_encoder : callable, optional
            Kept for API symmetry with ``__init__``. Never called on
            this path because the pool is already encoded.
        permutation_seed : int, optional
            When provided, each epoch's slice is reordered by a
            per-worker permutation seeded with this value mixed into
            the epoch counter. Preserves cross-worker diversity when
            many workers share the same pool contents.
        """
        manifest_path = os.path.join(output_dir, cls._MANIFEST_NAME)
        with open(manifest_path, "r", encoding="utf-8") as fd:
            manifest = json.load(fd)

        encoded_path = os.path.join(output_dir, manifest["encoded_file"])
        peptides_path = os.path.join(output_dir, manifest["peptides_file"])
        with open(peptides_path, "r", encoding="utf-8") as fd:
            peptides = json.load(fd)

        shape = tuple(manifest["shape"])
        pool_epochs = int(manifest["pool_epochs"])
        expected_total = int(manifest["total_count"])
        worker_total = int(planner.get_total_count())
        if expected_total != worker_total:
            raise ValueError(
                "Shared pool total_count=%d does not match worker "
                "planner total_count=%d — regenerate the pool or "
                "align the planner hyperparameters." % (
                    expected_total, worker_total,
                )
            )
        encoded = numpy.memmap(
            encoded_path, dtype=manifest["dtype"], mode="r", shape=shape
        )

        def _refuse_reencode(_encodable_sequences):
            raise RuntimeError(
                "Shared-mmap RandomNegativesPool only holds cycle 0. "
                "Training has advanced to a cycle >= pool_epochs=%d, but "
                "the mmap pool cannot regenerate (no encoder available). "
                "Size the shared pool's ``pool_epochs`` to at least the "
                "training run's ``max_epochs`` so cycle 0 covers every "
                "epoch, or switch this worker to an in-process "
                "RandomNegativesPool that can rebuild." % pool_epochs
            )

        instance = cls(
            planner=planner,
            peptide_encoder=peptide_encoder or _refuse_reencode,
            pool_epochs=pool_epochs,
            seed=None,  # content is already deterministic in the mmap
        )
        instance._current_peptides = peptides
        instance._current_encoded = encoded
        instance._current_cycle = 0  # single cycle — mmap covers it all
        instance._permutation_seed = permutation_seed
        # Pool-epoch cap the caller must respect. ``get_epoch_inputs``
        # checks this and raises with a useful message instead of
        # blowing up with the generic _refuse_reencode RuntimeError
        # when crossed.
        instance._mmap_max_epoch = pool_epochs - 1
        return instance

    def _epoch_permutation(self, epoch):
        seed = getattr(self, "_permutation_seed", None)
        if seed is None:
            return None
        rng = numpy.random.default_rng(
            numpy.random.SeedSequence(
                entropy=int(seed), spawn_key=(int(epoch),)
            )
        )
        return rng.permutation(self._total_count)


# --- Device-resident random-negative encoding -----------------------------
#
# The host path generates peptides by sampling AA letters via
# ``numpy.random.choice``, joining each row into a Python string, and then
# round-tripping the strings through ``EncodableSequences.create`` →
# ``sequences_to_fixed_length_index_encoded_array`` to recover integer
# indices. On a release-exact run that string round-trip dominated random-
# negative generation: ~22.5 M Python string allocations + 1.5 M joins per
# epoch.
#
# The helpers below skip the round-trip entirely. They sample integer
# indices directly via ``torch.multinomial`` against the canonical AA
# distribution and apply the same per-(allele, length) alignment math as
# the host path, writing into a pre-allocated device-resident int8 tensor.
# Output rows match ``planner.get_peptides()`` order: outer loop over
# pool epochs, then ``plan_df`` row-major iteration of (allele, length,
# count). Within a (allele, length) block, the row order is the order
# returned by torch.multinomial — not the same RNG stream as the numpy
# path, but the row layout (which (allele, length) sits where) is
# byte-identical.


_SUPPORTED_DEVICE_ALIGNMENTS = (
    "left_pad_centered_right_pad",
    "pad_middle",
)


def _place_indices_with_alignment(
        out_rows,
        block,
        length,
        alignment,
        max_length,
        left_edge=4,
        right_edge=4):
    """Write a ``(count, length)`` index block into ``out_rows`` in place.

    ``out_rows`` is a (count, encoded_length) tensor or array prefilled
    with the X index. Mirrors the per-length branch of
    :func:`EncodableSequences.sequences_to_fixed_length_index_encoded_array`
    for the supported alignment methods, but operates on already-sampled
    integer indices instead of peptide strings.

    Factored out so the device-aware encoder and host-side parity tests
    share one source of truth for the alignment math.
    """
    L = int(length)
    if alignment == "left_pad_centered_right_pad":
        if L < 1 or L > int(max_length):
            raise ValueError(
                "left_pad_centered_right_pad requires 1 <= length <= "
                "max_length=%d; got %d" % (max_length, L)
            )
        out_rows[:, :L] = block
        out_rows[:, -L:] = block
        center_left_padding = (int(max_length) - L) // 2
        center_left_offset = int(max_length) + center_left_padding
        out_rows[:, center_left_offset : center_left_offset + L] = block
        return
    if alignment == "pad_middle":
        min_length = int(left_edge) + int(right_edge)
        if L < min_length or L > int(max_length):
            raise ValueError(
                "pad_middle requires %d <= length <= max_length=%d; "
                "got %d" % (min_length, max_length, L)
            )
        middle_length = int(max_length) - int(left_edge) - int(right_edge)
        num_null = int(max_length) - L
        num_null_left = int(math.ceil(num_null / 2))
        num_middle_filled = middle_length - num_null
        middle_start = int(left_edge) + num_null_left
        out_rows[:, : int(left_edge)] = block[:, : int(left_edge)]
        out_rows[:, -int(right_edge) :] = block[:, -int(right_edge) :]
        if num_middle_filled > 0:
            out_rows[
                :, middle_start : middle_start + num_middle_filled
            ] = block[
                :, int(left_edge) : int(left_edge) + num_middle_filled
            ]
        return
    raise NotImplementedError(
        "Device-resident RN encoder supports alignment_method in %r; "
        "got %r." % (_SUPPORTED_DEVICE_ALIGNMENTS, alignment)
    )


def encoded_length_for_alignment(alignment, max_length):
    """Return the per-row encoded length for an alignment method.

    Mirrors the shape contract of
    :func:`EncodableSequences.sequences_to_fixed_length_index_encoded_array`.
    """
    if alignment == "left_pad_centered_right_pad":
        return 3 * int(max_length)
    if alignment == "pad_middle":
        return int(max_length)
    if alignment == "left_pad_right_pad":
        return 2 * int(max_length)
    if alignment in ("right_pad", "left_pad"):
        return int(max_length)
    raise NotImplementedError(
        "encoded_length_for_alignment: unknown alignment %r" % (alignment,)
    )


def aa_distribution_to_index_weights(distribution, device=None, dtype=None):
    """Convert a letter-indexed AA distribution to a torch index-weight vector.

    Inputs
    ------
    distribution : pandas.Series or None
        Letter→probability map (typically the planner's
        ``aa_distribution``). When ``None``, weights are uniform over
        the 20 common amino acids.
    device, dtype : forwarded to the result tensor. ``dtype`` defaults
        to ``torch.float32`` (multinomial requires float).

    Returns
    -------
    torch.Tensor of shape ``(len(amino_acid.AMINO_ACIDS),)`` — index 20
    (X) is always zero so X is never sampled.
    """
    import torch as _torch

    if dtype is None:
        dtype = _torch.float32
    weights = _torch.zeros(
        len(amino_acid.AMINO_ACIDS), dtype=dtype, device=device
    )
    if distribution is None:
        for letter in amino_acid.COMMON_AMINO_ACIDS:
            weights[amino_acid.AMINO_ACID_INDEX[letter]] = 1.0
    else:
        for letter, prob in distribution.items():
            idx = amino_acid.AMINO_ACID_INDEX[letter]
            if idx == amino_acid.X_INDEX:
                continue
            weights[idx] = float(prob)
    total = weights.sum()
    if float(total) <= 0.0:
        raise ValueError(
            "aa_distribution_to_index_weights: all weights are zero. "
            "Either supply a distribution covering common AAs or pass "
            "distribution=None for a uniform fallback."
        )
    weights = weights / total
    return weights


def encode_random_negatives_on_device(
        *,
        planner,
        pool_epochs,
        peptide_encoding,
        device,
        generator=None,
        dtype=None,
        out=None):
    """Sample + encode ``pool_epochs`` cycles of random negatives on device.

    Parameters
    ----------
    planner : RandomNegativePeptides
        Already had ``plan(...)`` called.
    pool_epochs : int
        Number of consecutive get_peptides() epochs to materialize. The
        result has ``pool_epochs * planner.get_total_count()`` rows.
    peptide_encoding : dict
        Subset of ``Class1NeuralNetwork.network_hyperparameter_defaults``
        that controls peptide encoding. Required keys:
        ``alignment_method``, ``max_length``. For ``pad_middle``,
        ``left_edge`` and ``right_edge`` are also consulted (default 4
        each, matching the historical defaults).
    device : torch.device or str
        Output tensor device. Sampling and alignment writes happen here
        (no host round-trip).
    generator : torch.Generator, optional
        When supplied, all multinomial draws are routed through it. The
        caller is responsible for seeding (the numpy / torch RNG streams
        are not bridged — for reproducible parity tests, seed both).
    dtype : torch.dtype, optional
        Output dtype. Default ``torch.int8`` (alphabet size 21 fits with
        room to spare).
    out : torch.Tensor, optional
        Pre-allocated output tensor of the correct shape, dtype, and
        device. When supplied, this function refills it in place — the
        usual pattern for the device-resident fit path, where the
        per-fit RN buffer is allocated once and reused per cycle.

    Returns
    -------
    torch.Tensor of shape ``(pool_epochs * total_count, encoded_length)``,
    dtype int8 by default, on ``device``. Rows are the per-(allele,
    length) blocks in canonical ``planner.plan_df`` row-major order,
    repeated ``pool_epochs`` times — same layout the host path produces
    after encoding ``planner.get_peptides()``.
    """
    import torch as _torch

    if planner.plan_df is None:
        raise ValueError(
            "encode_random_negatives_on_device: planner has no plan; "
            "call planner.plan(...) before encoding."
        )
    pool_epochs = max(int(pool_epochs), 1)
    if dtype is None:
        dtype = _torch.int8

    alignment = peptide_encoding["alignment_method"]
    max_length = int(peptide_encoding["max_length"])
    if alignment not in _SUPPORTED_DEVICE_ALIGNMENTS:
        raise NotImplementedError(
            "encode_random_negatives_on_device: alignment_method %r not "
            "supported on device. Falls back to host encoding via "
            "EncodableSequences." % (alignment,)
        )
    left_edge = int(peptide_encoding.get("left_edge", 4))
    right_edge = int(peptide_encoding.get("right_edge", 4))
    encoded_length = encoded_length_for_alignment(alignment, max_length)

    total_count = int(planner.get_total_count())
    expected_rows = pool_epochs * total_count
    expected_shape = (expected_rows, encoded_length)
    if out is None:
        out = _torch.full(
            expected_shape,
            int(amino_acid.X_INDEX),
            dtype=dtype,
            device=device,
        )
    else:
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                "out shape mismatch: expected %r, got %r" % (
                    expected_shape, tuple(out.shape),
                )
            )
        if out.dtype != dtype:
            raise ValueError(
                "out dtype mismatch: expected %r, got %r" % (dtype, out.dtype)
            )
        out.fill_(int(amino_acid.X_INDEX))

    weights = aa_distribution_to_index_weights(
        planner.aa_distribution, device=device
    )

    plan_df = planner.plan_df
    alleles = list(plan_df.index)
    length_columns = list(plan_df.columns)

    cursor = 0
    for _epoch in range(pool_epochs):
        for allele in alleles:
            for length in length_columns:
                num = int(plan_df.at[allele, length])
                if num <= 0:
                    continue
                L = int(length)
                flat = _torch.multinomial(
                    weights,
                    num_samples=num * L,
                    replacement=True,
                    generator=generator,
                )
                block = flat.view(num, L).to(dtype)
                _place_indices_with_alignment(
                    out[cursor : cursor + num],
                    block,
                    length=L,
                    alignment=alignment,
                    max_length=max_length,
                    left_edge=left_edge,
                    right_edge=right_edge,
                )
                cursor += num
    if cursor != expected_rows:
        raise AssertionError(
            "encode_random_negatives_on_device: wrote %d rows, expected "
            "%d (planner.get_total_count() drift?)" % (
                cursor, expected_rows,
            )
        )
    return out
