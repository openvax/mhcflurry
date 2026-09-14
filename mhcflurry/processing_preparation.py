"""Resumable, bounded preparation of scored processing-negative pools."""

import hashlib
import json
from pathlib import Path
import shutil
import time
from dataclasses import dataclass

import numpy
import pandas

from .common import derive_seed, positive_int_arg
from .experiment_archive import sha256_file
from .processing_matching import (
    IncompleteProcessingMatches, matched_training_data, validate_matched_training_data)
from .numeric_proteome import NumericCandidatePool, sequence_keys, MAX_PROCESSING_LENGTH
from .numeric_sequences import NumericSequences


@dataclass
class ScoredValues:
    """Prediction values and actual scoring time, excluding pipeline queue waits."""

    values: object
    seconds: float


def _scored_frame(unscored, response):
    if not isinstance(response, ScoredValues):
        raise TypeError("Sample workflows require timed ScoredValues responses")
    if isinstance(unscored, NumericCandidatePool):
        return unscored.with_scores(response.values)
    return unscored.assign(affinity_prediction=response.values)


def add_processing_preparation_args(parser):
    """Options for the adaptive, resumable processing-data command."""
    parser.add_argument("--resume", action="store_true",
                        help="Resume the same verified processing preparation output.")
    parser.add_argument("--resume-matching-dir",
                        help="Reuse scored samples from a prior matching-artifacts directory.")
    parser.add_argument("--max-expansion-rounds", type=positive_int_arg, default=8,
                        help="Maximum additional candidate rounds per unresolved sample.")
    parser.add_argument("--expansion-candidates-per-length", type=positive_int_arg, default=100000,
                        help="Additional draws per unresolved length per round (default 100000).")
    parser.add_argument("--preparation-pipeline-depth", type=positive_int_arg, default=3,
                        help="Maximum active samples per scoring worker (default 3; 1 disables overlap).")


def write_json_atomic(path, value):
    """Publish metadata only after the corresponding data files are closed."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv_atomic(path, frame):
    """Keep interrupted compression from looking like a completed artifact."""
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="bz2")
    temporary.replace(path)


def validate_pool_hits(frame, hits):
    """Require exactly the current ordered observed rows and finite scores."""
    columns = ["peptide", "protein_accession", "n_flank", "c_flank", "sample_id"]
    observed = frame.loc[frame.hit.fillna(0).eq(1), columns].fillna("").astype(str).reset_index(drop=True)
    expected = hits[columns].fillna("").astype(str).reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError("Cached candidate pool has different observed hit rows")
    affinities = pandas.to_numeric(frame.affinity_prediction, errors="raise")
    if not numpy.isfinite(affinities).all() or (affinities <= 0).any():
        raise ValueError("Cached candidate pool has invalid affinity scores")
    if not frame.sample_id.eq(str(hits.sample_id.iloc[0])).all():
        raise ValueError("Cached candidate pool mixes samples")


class SamplePreparation:
    """Immutable scored rounds plus a checksummed final matched sample."""

    def __init__(self, args, sample, hits):
        self.args = args
        self.sample = str(sample)
        self.hits = hits
        self.directory = Path(str(args.out) + ".matching")
        self.name = hashlib.sha256(self.sample.encode()).hexdigest()[:24]
        self.rounds = []

    def path(self, suffix):
        return self.directory / (self.name + suffix)

    def load_pool(self):
        """Recover committed rounds, or import a verified prior sample pool."""
        prior = getattr(self.args, "resume_matching_dir", None)
        if prior and Path(prior).resolve() != self.directory.resolve():
            source_dir = Path(prior)
            prior_rounds = sorted(source_dir.glob(self.name + ".round-*.json"))
            for index, metadata_path in enumerate(prior_rounds):
                record = json.loads(metadata_path.read_text())
                expected = self.name + ".round-%02d.csv.bz2" % index
                source = source_dir / expected
                if (record["round"] != index or record["file"] != expected or
                        sha256_file(source) != record["sha256"]):
                    raise ValueError("Corrupt prior processing candidate rounds")
                target_metadata = self.path(".round-%02d.json" % index)
                if target_metadata.exists():
                    if json.loads(target_metadata.read_text())["sha256"] != record["sha256"]:
                        raise ValueError("Prior processing rounds differ from local rounds")
                    continue
                target = self.directory / expected
                temporary = target.with_name(target.name + ".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(target)
                write_json_atomic(target_metadata, {**record, "origin": str(source)})
        metadata = sorted(self.directory.glob(self.name + ".round-*.json"))
        if metadata:
            frames = []
            for index, path in enumerate(metadata):
                record = json.loads(path.read_text())
                expected = self.name + ".round-%02d.csv.bz2" % index
                if record["round"] != index or record["file"] != expected:
                    raise ValueError("Invalid or noncontiguous saved processing rounds")
                source = self.directory / expected
                if sha256_file(source) != record["sha256"]:
                    raise ValueError("Corrupt saved processing candidate scores")
                frames.append(pandas.read_csv(source, dtype={"sample_id": str}, low_memory=False))
                self.rounds.append(record)
            frame = pandas.concat(frames, ignore_index=True)
            validate_pool_hits(frame, self.hits)
            return frame
        # Old runs saved a single complete scored pool but no per-round hashes.
        source_dir = Path(prior) if prior else self.directory
        source = source_dir / (self.name + ".candidate_pool.csv.bz2")
        if not source.exists():
            return None
        frame = pandas.read_csv(source, dtype={"sample_id": str}, low_memory=False)
        validate_pool_hits(frame, self.hits)
        self.save_round(frame, {"round": 0, "origin": str(source),
            "origin_sha256": sha256_file(source), "origin_had_round_checksums": False,
            "sampler": "legacy-unversioned", "seed": None,
            "lengths": sorted(frame.peptide.str.len().unique().tolist())}, source=source)
        return frame

    def save_round(self, frame, record, source=None):
        index = len(self.rounds)
        path = self.path(".round-%02d.csv.bz2" % index)
        started = time.monotonic()
        if source is None:
            write_csv_atomic(path, frame)
        else:
            temporary = path.with_name(path.name + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(path)
        record.update(round=index, file=path.name, sha256=sha256_file(path),
                      sample_id=self.sample, rows=len(frame),
                      artifact_write_seconds=time.monotonic() - started)
        write_json_atomic(self.path(".round-%02d.json" % index), record)
        self.rounds.append(record)
        if index == 0:
            # Retain the established single-pool artifact name for consumers.
            shutil.copy2(path, self.path(".candidate_pool.csv.bz2"))

    def load_result(self):
        marker = self.path(".complete.json")
        if not marker.exists():
            return None
        record = json.loads(marker.read_text())
        path = self.path(".matched.csv.bz2")
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Corrupt saved matched processing sample")
        if record["round_sha256"] != [r["sha256"] for r in self.rounds]:
            raise ValueError("Matched processing sample refers to changed rounds")
        frame = pandas.read_csv(path, dtype={"sample_id": str}, low_memory=False)
        validate_pool_hits(frame, self.hits)
        validate_matched_training_data(frame)
        return frame

    def finish(self, frame, diagnostics):
        diagnostics.update(sample_id=self.sample, rounds=len(self.rounds))
        path = self.path(".matched.csv.bz2")
        write_csv_atomic(path, frame)
        write_json_atomic(self.path(".matching.json"), diagnostics)
        write_json_atomic(self.path(".complete.json"), {
            "sha256": sha256_file(path), "sample_id": self.sample,
            "round_sha256": [r["sha256"] for r in self.rounds]})


def sample_preparation_steps(args, sample, hits, seed, initial_pool, additional_candidates):
    """CPU coroutine yielding unscored peptides and receiving affinity scores.

    Callbacks keep proteome access and affinity model ownership in the worker.
    Initial/additional pools contain unscored rows. Yielded requests contain only
    peptides absent from committed rounds. Each continuation commits its scored
    round before matching, expanding, or publishing sample completion.
    """
    store = SamplePreparation(args, sample, hits)
    pool = store.load_pool()
    if pool is not None:
        saved = store.load_result()
        if saved is not None:
            print("Reusing completed matched sample", sample, flush=True)
            return saved
        print("Reusing scored pool", sample, len(pool), flush=True)
    else:
        started = time.monotonic()
        unscored = initial_pool()
        sampled_at = time.monotonic()
        response = yield unscored if isinstance(unscored, NumericCandidatePool) else unscored.peptide
        materializing_at = time.monotonic()
        pool = _scored_frame(unscored, response)
        store.save_round(pool, {"seed": seed, "sampler": "numeric-positions-v1",
            "lengths": sorted(pool.peptide.str.len().unique().tolist()),
            "sampling_seconds": sampled_at - started, "scoring_seconds": response.seconds,
            "materialization_seconds": time.monotonic() - materializing_at,
            "score_queue_seconds": max(0, materializing_at - sampled_at - response.seconds)})
    while True:
        started = time.monotonic()
        try:
            result, diagnostics = matched_training_data(pool, args.matching_reference,
                decoys_per_hit=args.decoys_per_hit, max_distance=args.max_affinity_distance)
        except IncompleteProcessingMatches as error:
            failed_at = time.monotonic()
            next_round = len(store.rounds)
            lengths = sorted({failure["peptide_length"] for failure in error.failures})
            request = {"sample_id": str(sample), "round": next_round,
                       "unresolved_hits": error.failures, "lengths": lengths,
                       "matching_seconds": failed_at - started}
            write_json_atomic(store.path(".failure.json"), request)
            if next_round > args.max_expansion_rounds:
                raise IncompleteProcessingMatches(
                    "Processing candidate expansion budget exhausted: " + str(error), error.failures) from error
            parts = []
            seeds = {}
            # One exclusion set serves every length: the pool is fixed for the
            # whole round, so rebuilding it per length only repeats the work.
            scored_peptides = set(pool.peptide)
            for length in lengths:
                round_seed = derive_seed(seed, "processing-expansion-v1", next_round, length)
                seeds[str(length)] = round_seed
                parts.append(additional_candidates(length, scored_peptides, round_seed))
            if isinstance(parts[0], NumericCandidatePool):
                from .numeric_proteome import ProteinWindows
                if any(not part.hits.empty for part in parts):
                    raise ValueError("Expansion may not add observed hits")
                windows = ProteinWindows.concatenate([part.windows for part in parts])
                # Not redundant with the exclusion set above: samplers are
                # caller-supplied, so committed peptides are rejected here
                # numerically instead of trusting the callback to honor it.
                known = sequence_keys(NumericSequences.from_strings(pool.peptide, MAX_PROCESSING_LENGTH).indices.numpy())
                windows = windows.take(numpy.flatnonzero(~numpy.isin(windows.keys, known)))
                unscored = NumericCandidatePool(parts[0].hits, windows, sample, parts[0].flank_length)
                count = len(windows)
                valid_lengths = numpy.isin(windows.lengths, lengths).all()
            else:
                unscored = pandas.concat(parts, ignore_index=True)
                unscored = unscored.loc[~unscored.peptide.isin(pool.peptide)].drop_duplicates("peptide").copy()
                count = len(unscored)
                valid_lengths = unscored.peptide.str.len().isin(lengths).all()
            if not count:
                raise IncompleteProcessingMatches(
                    "No unscored candidates remain in unresolved processing pools: " + str(error), error.failures) from error
            if not valid_lengths:
                raise ValueError("Expansion generated an unrelated peptide length")
            sampled_at = time.monotonic()
            response = yield unscored if isinstance(unscored, NumericCandidatePool) else unscored.peptide
            materializing_at = time.monotonic()
            additions = _scored_frame(unscored, response)
            store.save_round(additions, {**request, "seeds_by_length": seeds,
                "sampler": "numeric-positions-v1",
                "sampling_seconds": sampled_at - failed_at, "scoring_seconds": response.seconds,
                "materialization_seconds": time.monotonic() - materializing_at,
                "score_queue_seconds": max(0, materializing_at - sampled_at - response.seconds)})
            pool = pandas.concat([pool, additions], ignore_index=True)
            print("Expanded", sample, "round", next_round, "lengths", lengths,
                  "new scored peptides", len(additions), flush=True)
        else:
            diagnostics["matching_seconds"] = time.monotonic() - started
            store.finish(result, diagnostics)
            return result


def prepare_sample(args, sample, hits, seed, initial_pool, additional_candidates, score):
    """Synchronous adapter for the same resumable sample workflow."""
    workflow = sample_preparation_steps(args, sample, hits, seed, initial_pool, additional_candidates)
    value = None
    try:
        while True:
            try:
                request = workflow.send(value)
            except StopIteration as finished:
                return finished.value
            started = time.monotonic()
            values = score(request)
            value = ScoredValues(values, time.monotonic() - started)
    finally:
        workflow.close()
