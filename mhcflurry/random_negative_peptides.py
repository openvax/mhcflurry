import json
import logging
import math
import os

import numpy
import pandas

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

    def __init__(self, planner, peptide_encoder, pool_epochs=1, seed=None):
        """
        Parameters
        ----------
        planner : RandomNegativePeptides
            A planner that has already had ``plan(...)`` called on it.

        peptide_encoder : callable
            Receives an ``EncodableSequences`` and returns an encoded
            numpy array (typically ``Class1NeuralNetwork
            .peptides_to_network_input``). Called once per cycle on the
            full pool.

        pool_epochs : int
            Number of consecutive epochs that share a pool. 1 is
            semantically identical to pre-Phase-1 behavior.

        seed : int, optional
            Seed for the per-cycle RNG. When None, draws go through
            numpy's global state (same as pre-Phase-1).
        """
        assert planner.plan_df is not None, "Call planner.plan() first"
        self.planner = planner
        self.peptide_encoder = peptide_encoder
        self.pool_epochs = max(int(pool_epochs), 1)
        self.seed = seed
        self._total_count = int(planner.get_total_count())
        self._current_cycle = None
        self._current_encoded = None  # shape: (pool_epochs * total_count, ...)
        self._current_peptides = None  # list of str, len = pool_epochs * total_count

    def _rng_for_cycle(self, cycle):
        if self.seed is None:
            return None
        seed_seq = numpy.random.SeedSequence(
            entropy=int(self.seed), spawn_key=(int(cycle),)
        )
        return numpy.random.default_rng(seed_seq)

    def _build_cycle(self, cycle):
        rng = self._rng_for_cycle(cycle)
        peptides = []
        for _ in range(self.pool_epochs):
            peptides.extend(self.planner.get_peptides(rng=rng))
        assert len(peptides) == self.pool_epochs * self._total_count
        encoded = self.peptide_encoder(EncodableSequences.create(peptides))
        self._current_peptides = peptides
        self._current_encoded = encoded
        self._current_cycle = cycle

    def get_epoch_inputs(self, epoch):
        """Return ``(peptides_list, encoded_slice)`` for ``epoch``.

        The returned ``encoded_slice`` is a view into the pool-level
        encoded array — no copy — so downstream assignments that expect
        to own the memory should ``.copy()`` it first. The list of raw
        peptide strings is included for callers that need it (e.g.
        logging / diagnostics).

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
        peptides_slice = self._current_peptides[offset:end]
        encoded_slice = self._current_encoded[offset:end]
        permutation = self._epoch_permutation(epoch)
        if permutation is not None:
            peptides_slice = [peptides_slice[i] for i in permutation]
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
    # before it forks workers) encode the pool once and persist it as
    # an int8 memmap + JSON manifest. Workers then load the pool with
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
        builder._build_cycle(0)
        encoded = numpy.asarray(builder._current_encoded)
        peptides = list(builder._current_peptides)

        os.makedirs(output_dir, exist_ok=True)
        encoded_path = os.path.join(output_dir, cls._ENCODED_NAME)
        peptides_path = os.path.join(output_dir, cls._PEPTIDES_NAME)
        manifest_path = os.path.join(output_dir, cls._MANIFEST_NAME)

        encoded_int8 = encoded.astype("int8", copy=False)
        mm = numpy.memmap(
            encoded_path, dtype="int8", mode="w+", shape=encoded_int8.shape
        )
        mm[:] = encoded_int8
        mm.flush()
        del mm

        with open(peptides_path, "w", encoding="utf-8") as fd:
            json.dump(peptides, fd)

        manifest = {
            "shape": list(encoded_int8.shape),
            "dtype": "int8",
            "pool_epochs": int(pool_epochs),
            "total_count": int(builder._total_count),
            "seed": int(seed),
            "encoded_file": cls._ENCODED_NAME,
            "peptides_file": cls._PEPTIDES_NAME,
        }
        with open(manifest_path, "w", encoding="utf-8") as fd:
            json.dump(manifest, fd, indent=2, sort_keys=True)
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
