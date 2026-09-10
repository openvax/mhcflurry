"""Shared, auditable affinity/length matching for processing data."""

import hashlib
import json
from pathlib import Path

import numpy
import pandas

from .experiment_archive import sha256_file
from .common import positive_int_arg, positive_float_arg


MATCHING_POLICY = "sample-length-affinity-v1"


class IncompleteProcessingMatches(ValueError):
    """A valid scored pool lacks enough negatives for the listed hit rows."""

    def __init__(self, message, failures):
        super().__init__(message)
        self.failures = failures

    def __reduce__(self):
        return type(self), (str(self), self.failures)


def add_processing_matching_args(parser):
    """Shared processing-data CLI policy; legacy filtering is opt-in."""
    parser.add_argument("--negative-policy", choices=("matched", "legacy-top-binders"), default="matched")
    parser.add_argument("--decoys-per-hit", type=positive_int_arg, default=1,
                        help="Matched training negatives per hit (default 1).")
    parser.add_argument("--max-affinity-distance", type=positive_float_arg, default=0.25,
                        help="Maximum log10-affinity distance, at most 0.25.")


def initialize_matching_artifacts(args):
    """Fingerprint the reference and data before generating any matched table."""
    if args.negative_policy == "legacy-top-binders":
        return None
    if not numpy.isfinite(args.max_affinity_distance) or not 0 < args.max_affinity_distance <= 0.25:
        raise ValueError("Processing matching caliper may not exceed 0.25 log10 units")
    reference = affinity_reference_fingerprint(args.affinity_predictor)
    directory = Path(str(args.out) + ".matching")
    if (directory.exists() or Path(args.out).exists()) and not getattr(args, "resume", False):
        raise ValueError("Use fresh processing data output; preserve prior matching artifacts")
    sources = [args.hits, args.proteome_peptides or args.proteome_reference_csv]
    if getattr(args, "exclude_samples_file", None):
        sources.append(args.exclude_samples_file)
    provenance = {"policy": MATCHING_POLICY, "affinity_reference": reference,
                  "decoys_per_hit": args.decoys_per_hit,
                  "max_log10_affinity_distance": args.max_affinity_distance,
                  "candidate_pool_multiplier": args.ppv_multiplier,
                  "seed": getattr(args, "random_seed", None),
                  "inputs": {str(Path(p).resolve()): sha256_file(p) for p in sources}}
    policy_path = directory / "experiment.json"
    if getattr(args, "resume", False):
        if not policy_path.is_file():
            raise ValueError("Cannot resume without saved matching provenance")
        saved = json.loads(policy_path.read_text())
        validate_preparation_provenance(saved, provenance)
        if not getattr(args, "resume_matching_dir", None):
            args.resume_matching_dir = saved.get("reused_matching_artifacts", {}).get("path")
    prior = getattr(args, "resume_matching_dir", None)
    if prior:
        prior_path = Path(prior) / "experiment.json"
        validate_preparation_provenance(json.loads(prior_path.read_text()), provenance)
        provenance["reused_matching_artifacts"] = {
            "path": str(Path(prior).resolve()), "experiment_sha256": sha256_file(prior_path)}
        if getattr(args, "resume", False) and saved.get("reused_matching_artifacts", {}).get(
                "experiment_sha256") != sha256_file(prior_path):
            raise ValueError("Changed processing preparation cache provenance")
    directory.mkdir(parents=True, exist_ok=True)
    if not policy_path.exists():
        provenance.update(schema_version=2, sampler="numeric-positions-v1")
        if hasattr(args, "preparation_pipeline_depth"):
            provenance.update(preparation_pipeline_depth=args.preparation_pipeline_depth,
                candidate_representation="protein-indices-v1" if args.proteome_reference_csv else "dataframe",
                matching_implementation="batched-stable-local-v1")
        policy_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return reference


def validate_preparation_provenance(saved, current):
    """Allow relocated caches only when scientific inputs and policy agree."""
    for key in ("policy", "decoys_per_hit", "max_log10_affinity_distance",
                "candidate_pool_multiplier", "seed"):
        if saved.get(key) != current.get(key):
            raise ValueError("Changed processing preparation input: " + key)
    if saved["affinity_reference"]["sha256"] != current["affinity_reference"]["sha256"]:
        raise ValueError("Changed processing preparation affinity reference")
    def normalized_inputs(value):
        pairs = [(Path(path).name, digest) for path, digest in value["inputs"].items()]
        if len({name for name, _ in pairs}) != len(pairs):
            raise ValueError("Ambiguous preparation input basenames")
        return sorted(pairs)
    if normalized_inputs(saved) != normalized_inputs(current):
        raise ValueError("Changed processing preparation input files")


def prepare_processing_training_sample(frame, args, reference):
    """Save the scored pool before matching; retain failure evidence too."""
    directory = Path(str(args.out) + ".matching")
    sample = str(frame.sample_id.iloc[0])
    name = hashlib.sha256(sample.encode()).hexdigest()[:24]
    frame.to_csv(directory / (name + ".candidate_pool.csv.bz2"), index=False)
    try:
        matched, diagnostics = matched_training_data(
            frame, reference, decoys_per_hit=args.decoys_per_hit,
            max_distance=args.max_affinity_distance)
    except ValueError as error:
        (directory / (name + ".failure.json")).write_text(json.dumps(
            {"sample_id": sample, "error": str(error)}, indent=2) + "\n")
        raise
    diagnostics["sample_id"] = sample
    (directory / (name + ".matching.json")).write_text(json.dumps(diagnostics, indent=2) + "\n")
    return matched


def _sorted_pool(frame, indices):
    indices = numpy.asarray(indices, dtype="int64")
    order = numpy.argsort(
        frame.log10_affinity.to_numpy()[indices], kind="stable")
    indices = indices[order]
    return (
        indices,
        frame.log10_affinity.to_numpy()[indices],
    )


def _nearest(pool, target, count, excluded=(), max_distance=None):
    indices, values = pool
    if count <= 0 or not len(indices):
        return []
    position = int(numpy.searchsorted(values, target))
    radius = min(len(indices), max(count * 3, count + len(excluded)))
    start = max(0, position - radius)
    end = min(len(indices), position + radius)
    candidates = indices[start:end]
    candidate_values = values[start:end]
    order = numpy.argsort(numpy.abs(candidate_values - target), kind="stable")
    excluded = set(excluded)
    selected = []
    for offset in order:
        index = int(candidates[offset])
        if (
                max_distance is not None and
                abs(float(candidate_values[offset]) - target) > max_distance):
            continue
        if index not in excluded:
            selected.append(index)
            if len(selected) == count:
                break
    if len(selected) < count and len(candidates) < len(indices):
        # A very dense set of excluded near-neighbors can exhaust the local
        # slice. The full stable ordering is the deterministic fallback.
        order = numpy.argsort(numpy.abs(values - target), kind="stable")
        for offset in order:
            index = int(indices[offset])
            if (
                    max_distance is not None and
                    abs(float(values[offset]) - target) > max_distance):
                continue
            if index not in excluded and index not in selected:
                selected.append(index)
                if len(selected) == count:
                    break
    return selected


def nearest_affinity_batch(pool, targets, count, excluded=None,
                           max_distance=None, chunk_size=4096):
    """Match many targets to one sorted pool, with scalar-compatible ties.

    Parameters
    ----------
    pool : tuple of numpy.ndarray
        Original row indices and ascending float64 affinities (stable order).
    targets : array-like
        Target log10 affinities.
    count : int
        Maximum matches per target.
    excluded : numpy.ndarray, optional
        Per-target excluded row indices, padded with -1.
    max_distance : float, optional
        Inclusive affinity caliper.
    chunk_size : int
        Maximum targets per temporary search matrix.

    Returns
    -------
    numpy.ndarray
        Original row indices, shape (len(targets), count), padded with -1.
        Ties use the historical local search window, not a global tie order.
    """
    indices, values = pool
    targets = numpy.asarray(targets, dtype="float64")
    return _nearest_windows(indices, values, targets, count, excluded, max_distance,
        chunk_size, numpy.searchsorted(values, targets),
        numpy.zeros(len(targets), dtype="int64"), numpy.full(len(targets), len(indices)))


def _nearest_windows(indices, values, targets, count, excluded, max_distance,
                     chunk_size, positions, lower_bounds, upper_bounds):
    """Shared bounded window selection for single or packed sorted pools."""
    if count < 0 or chunk_size < 1:
        raise ValueError("Invalid nearest-affinity batch size")
    result = numpy.full((len(targets), count), -1, dtype="int64")
    if not count or not len(indices):
        return result
    excluded = (numpy.empty((len(targets), 0), dtype="int64") if excluded is None
                else numpy.asarray(excluded, dtype="int64"))
    if excluded.ndim != 2 or len(excluded) != len(targets):
        raise ValueError("Exclusions must have one row per target")
    radius = numpy.minimum(upper_bounds - lower_bounds, numpy.maximum(
        count * 3, count + (excluded >= 0).sum(axis=1)))
    # Bound temporary memory even for unusually large requested risk sets.
    width = 2 * int(radius.max()) if len(radius) else 0
    chunk_size = min(chunk_size, max(1, 1_000_000 // max(1, width)))
    for start in range(0, len(targets), chunk_size):
        end = min(len(targets), start + chunk_size)
        target = targets[start:end]
        position = positions[start:end]
        offsets = numpy.arange(-int(radius[start:end].max()), int(radius[start:end].max()))
        locations = position[:, None] + offsets
        valid = ((locations >= lower_bounds[start:end, None]) & (locations < upper_bounds[start:end, None]) &
                 (offsets >= -radius[start:end, None]) & (offsets < radius[start:end, None]))
        safe = locations.clip(0, len(indices) - 1)
        candidates = indices[safe]
        distance = numpy.abs(values[safe].astype("float64", copy=False) - target[:, None])
        sort_dtype = numpy.result_type(values.dtype, 0.0)
        sort_distance = (distance if sort_dtype == numpy.dtype("float64") else
                         numpy.abs(values[safe] - target.astype(sort_dtype)[:, None]))
        if max_distance is not None:
            valid &= distance <= max_distance
        for column in excluded[start:end].T:
            valid &= candidates != column[:, None]
        order = numpy.argsort(numpy.where(valid, sort_distance, numpy.inf), axis=1, kind="stable")[:, :count]
        selected = numpy.take_along_axis(candidates, order, axis=1)
        selected_valid = numpy.take_along_axis(valid, order, axis=1)
        result[start:end, :selected.shape[1]] = numpy.where(selected_valid, selected, -1)
        # Preserve the reference's exceptional full-pool fallback. A failed
        # caliper lookup cannot find anything further away outside the window.
        for local in numpy.flatnonzero((result[start:end] >= 0).sum(axis=1) < count):
            begin, finish = lower_bounds[start + local], upper_bounds[start + local]
            lower = max(begin, int(position[local] - radius[start + local]))
            upper = min(finish, int(position[local] + radius[start + local]))
            outside = ([abs(values[lower - 1] - target[local])] if lower > begin else [])
            if upper < finish:
                outside.append(abs(values[upper] - target[local]))
            if not outside or (max_distance is not None and min(outside) > max_distance):
                continue
            omitted = excluded[start + local]
            selected = _nearest((indices[begin:finish], values[begin:finish]), target[local], count,
                                omitted[omitted >= 0], max_distance)
            result[start + local, :len(selected)] = selected
    return result


class SortedAffinityPools:
    """One stable numeric index for many sample/length[/protein] pools."""

    def __init__(self, frame, negative_indices, columns):
        codes = frame.groupby(columns, sort=False, dropna=False).ngroup().to_numpy()
        self.codes = codes.astype("int64")
        indices = numpy.asarray(negative_indices, dtype="int64")
        missing = frame[columns].isna().any(axis=1)
        if missing.any():
            # Preserve the old tuple-key behavior even for missing identities:
            # pandas string NaN, object None and pd.NA need not compare alike.
            # Only this exceptional path uses scalar row/key lookup.
            legacy_keys = {key: int(self.codes[group.index[0]])
                for key, group in frame.loc[indices].groupby(
                    columns, sort=False, dropna=False)}
            for row_index, row in frame.loc[missing & frame.hit.eq(1)].iterrows():
                self.codes[row_index] = legacy_keys.get(tuple(row[c] for c in columns), -1)
        indices = indices[self.codes[indices] >= 0]
        values = frame.log10_affinity.to_numpy()
        order = numpy.lexsort((values[indices], self.codes[indices]))
        self.indices = indices[order]
        self.values = values[self.indices]
        group_ids = self.codes[self.indices]
        counts = numpy.bincount(group_ids, minlength=max(0, int(self.codes.max()) + 1))
        self.ends = numpy.cumsum(counts)
        self.starts = self.ends - counts
        self.keys = numpy.empty(len(indices), dtype=[("group", "int64"), ("value", "float64")])
        self.keys["group"] = group_ids
        self.keys["value"] = self.values

    def nearest(self, hit_indices, targets, count, excluded=None, max_distance=None):
        """Batched lookup across all groups; -1 denotes a missing match."""
        groups = self.codes[hit_indices]
        valid = groups >= 0
        result = numpy.full((len(groups), count), -1, dtype="int64")
        if not valid.any() or not len(self.indices):
            return result
        groups = groups[valid]
        targets = numpy.asarray(targets, dtype="float64")[valid]
        queries = numpy.empty(len(targets), dtype=self.keys.dtype)
        queries["group"], queries["value"] = groups, targets
        result[valid] = _nearest_windows(self.indices, self.values, targets, count,
            None if excluded is None else excluded[valid], max_distance, 4096,
            numpy.searchsorted(self.keys, queries), self.starts[groups], self.ends[groups])
        return result


def make_affinity_controlled_risk_sets(
        frame, decoys_per_hit=10, same_protein_caliper=0.25,
        max_distance=0.25):
    """Return hit-centered risk sets with nearest-affinity decoys."""
    if decoys_per_hit < 1 or int(decoys_per_hit) != decoys_per_hit:
        raise ValueError("decoys_per_hit must be positive")
    if not numpy.isfinite(max_distance) or max_distance < 0:
        raise ValueError("Matching requires a finite nonnegative affinity caliper")
    if frame.empty or not frame.index.equals(pandas.RangeIndex(len(frame))):
        raise ValueError("Matching requires a nonempty, range-indexed cohort")
    if not numpy.isfinite(frame.log10_affinity).all():
        raise ValueError("Matching affinity must be finite")
    if frame[["sample_id", "peptide", "peptide_len", "hit"]].isna().any().any():
        raise ValueError("Matching identities and labels must be nonmissing")
    if not frame.hit.isin([0, 1]).all() or not frame.hit.eq(1).any():
        raise ValueError("Matching requires binary labels and at least one hit")
    if not frame.peptide.str.len().eq(frame.peptide_len).all():
        raise ValueError("Peptide lengths do not match sequences")
    if same_protein_caliper is not None and (
            not numpy.isfinite(same_protein_caliper) or same_protein_caliper < 0):
        raise ValueError("same_protein_caliper must be finite and nonnegative")
    negatives = frame.index[frame.hit == 0].to_numpy(dtype="int64")
    # One sequence per sample is one decoy, even if it maps to many proteins.
    # Factorize unsorted identities once. Building two MultiIndexes sorted
    # hundreds of thousands of peptide strings just to check set membership.
    identities = frame.groupby(["sample_id", "peptide"], sort=False).ngroup().to_numpy(dtype="int64")
    if numpy.isin(identities[negatives], identities[frame.hit == 1]).any():
        raise ValueError("An observed peptide is also labelled as a negative in its sample")
    _, first = numpy.unique(identities[negatives], return_index=True)
    negative_frame = frame.loc[negatives[numpy.sort(first)]]
    global_pools = SortedAffinityPools(frame, negative_frame.index, ["sample_id", "peptide_len"])
    protein_pools = SortedAffinityPools(frame, negative_frame.index, ["sample_id", "peptide_len", "protein_accession"])

    hit_indices = frame.index[frame.hit == 1].to_numpy()
    hits = frame.loc[hit_indices].reset_index(drop=True)
    targets = hits.log10_affinity.to_numpy(dtype="float64")
    protein_distance = (min(max_distance, same_protein_caliper)
                        if same_protein_caliper is not None else max_distance)
    selected = protein_pools.nearest(hit_indices, targets, decoys_per_hit, max_distance=protein_distance)
    same_counts = (selected >= 0).sum(axis=1)
    fallback_count = 0
    for needed in numpy.unique(decoys_per_hit - same_counts):
        if not needed:
            continue
        rows = numpy.flatnonzero(decoys_per_hit - same_counts == needed)
        fallback = global_pools.nearest(hit_indices[rows], targets[rows], int(needed),
                                        excluded=selected[rows], max_distance=max_distance)
        selected[rows[:, None], same_counts[rows, None] + numpy.arange(needed)] = fallback
        fallback_count += int((fallback >= 0).sum())
    counts = (selected >= 0).sum(axis=1)
    unresolved = numpy.flatnonzero(counts != decoys_per_hit)
    if len(unresolved):
        failures = [{"sample_id": str(hits.at[i, "sample_id"]), "peptide": hits.at[i, "peptide"],
                     "available": int(counts[i]), "log10_affinity": float(targets[i]),
                     "peptide_length": int(hits.at[i, "peptide_len"])} for i in unresolved]
        raise IncompleteProcessingMatches(
            "%d hits could not be assigned %d affinity/length-matched decoys "
            "within %.3g log10 units. Expand the scored candidate pool; "
            "no hits were silently dropped and no unmatched fallback was used. Examples: %s" % (
                len(unresolved), decoys_per_hit, max_distance, failures[:5]), failures)
    row_indices = numpy.column_stack([hit_indices, selected]).ravel()
    risk_ids = numpy.repeat(numpy.arange(len(hits)), decoys_per_hit + 1)
    match_ranks = numpy.tile(numpy.arange(decoys_per_hit + 1), len(hits))
    same_protein = (numpy.arange(decoys_per_hit + 1)[None, :] <= same_counts[:, None]).ravel()
    distances = numpy.abs(frame.log10_affinity.to_numpy()[row_indices] - targets[risk_ids])
    result = frame.loc[row_indices].copy().reset_index().rename(
        columns={"index": "source_row"})
    result["risk_set_id"] = numpy.asarray(risk_ids, dtype="int64")
    result["match_rank"] = numpy.asarray(match_ranks, dtype="int16")
    result["same_protein_match"] = numpy.asarray(same_protein, dtype=bool)
    result["log10_affinity_distance"] = numpy.asarray(
        distances, dtype="float64")
    diagnostics = {
        "policy": MATCHING_POLICY,
        "max_log10_affinity_distance": max_distance,
        "risk_sets": int(result.risk_set_id.nunique()),
        "rows": int(len(result)),
        "decoys_per_hit": int(decoys_per_hit),
        "same_protein_caliper": same_protein_caliper,
        "fallback_decoys": int(fallback_count),
        "same_protein_decoy_fraction": float(
            result.loc[result.hit == 0, "same_protein_match"].mean()),
        "median_log10_affinity_distance": float(
            result.loc[result.hit == 0, "log10_affinity_distance"].median()),
        "p95_log10_affinity_distance": float(
            result.loc[result.hit == 0, "log10_affinity_distance"].quantile(.95)),
    }
    return result, diagnostics


def affinity_reference_fingerprint(directory):
    """Identify the frozen reference using its manifest, weights and sequences."""
    directory = Path(directory)
    paths = [directory / "manifest.csv", directory / "allele_sequences.csv"]
    for name in pandas.read_csv(paths[0], usecols=["model_name"]).model_name:
        if Path(name).name != name:
            raise ValueError("Invalid affinity model name")
        paths.append(directory / ("weights_" + name + ".npz"))
    files = {path.name: sha256_file(path) for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"path": str(directory.resolve()), "sha256": digest, "files": files}


def matched_training_data(frame, reference, decoys_per_hit=1, max_distance=0.25):
    """Retain hits and construct a fixed number of matched negatives per hit."""
    frame = frame.copy().reset_index(drop=True)
    if not frame.hit.fillna(0).isin([0, 1]).all():
        raise ValueError("Matching requires binary hit labels")
    frame["hit"] = frame.hit.fillna(0).astype(int)
    affinity = pandas.to_numeric(frame.affinity_prediction, errors="raise")
    if not numpy.isfinite(affinity).all() or (affinity <= 0).any():
        raise ValueError("Matching affinity must be finite and positive")
    frame["log10_affinity"] = numpy.log10(affinity)
    frame["peptide_len"] = frame.peptide.str.len()
    matched, diagnostics = make_affinity_controlled_risk_sets(
        frame, decoys_per_hit=decoys_per_hit, max_distance=max_distance)
    matched["processing_matching_policy"] = MATCHING_POLICY
    matched["matching_affinity_reference_sha256"] = reference["sha256"]
    matched["matching_max_log10_distance"] = max_distance
    matched["matching_decoys_per_hit"] = decoys_per_hit
    return matched, diagnostics


def validate_matched_training_data(frame, policy="matched"):
    """Reject unmatched/corrupt cached tables before training or resuming."""
    if policy == "legacy":
        return
    if policy != "matched":
        raise ValueError("Unknown processing data policy: " + policy)
    required = ["sample_id", "peptide", "hit", "risk_set_id", "affinity_prediction",
                "processing_matching_policy", "matching_affinity_reference_sha256",
                "matching_max_log10_distance", "matching_decoys_per_hit"]
    if frame.empty or set(required) - set(frame):
        raise ValueError("Processing requires matched training data. Regenerate it; "
                         "use --processing-data-policy legacy only for explicit historical replay.")
    if frame[required].isna().any().any() or not frame.processing_matching_policy.eq(MATCHING_POLICY).all():
        raise ValueError("Invalid processing matching metadata")
    for name in ("matching_affinity_reference_sha256", "matching_max_log10_distance", "matching_decoys_per_hit"):
        if frame[name].nunique() != 1:
            raise ValueError("Inconsistent matching configuration: " + name)
    caliper = float(frame.matching_max_log10_distance.iloc[0])
    count = float(frame.matching_decoys_per_hit.iloc[0])
    reference = str(frame.matching_affinity_reference_sha256.iloc[0])
    if len(reference) != 64 or any(c not in "0123456789abcdef" for c in reference):
        raise ValueError("Invalid affinity reference fingerprint")
    validate_matching_assignments(frame, "affinity_prediction", count, caliper)
    fold_columns = [c for c in frame if c.startswith("fold_") and c[5:].isdigit()]
    if fold_columns and frame.groupby("sample_id")[fold_columns].nunique().gt(1).any().any():
        raise ValueError("Matched training folds must keep complete samples together")


def validate_matching_assignments(frame, affinity_column, count, caliper=0.25):
    """Recompute matching invariants instead of trusting saved distances."""
    count = float(count)
    if not numpy.isfinite(caliper) or not 0 <= caliper <= 0.25 or count < 1 or not count.is_integer():
        raise ValueError("Invalid matching caliper or negative count")
    required = ["sample_id", "risk_set_id", "peptide", "hit", affinity_column]
    if frame.empty or set(required) - set(frame) or frame[required].isna().any().any():
        raise ValueError("Matching assignments are empty or missing required identities")
    values = pandas.to_numeric(frame[affinity_column], errors="raise")
    if not numpy.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid matching affinities")
    groups = frame.groupby(["sample_id", "risk_set_id"], sort=False)
    sizes = groups.hit.agg(["sum", "count"])
    if not frame.hit.isin([0, 1]).all() or not sizes["sum"].eq(1).all() or not sizes["count"].eq(count + 1).all():
        raise ValueError("Invalid matched hit/decoy ratio")
    lengths = frame.assign(_length=frame.peptide.str.len()).groupby(["sample_id", "risk_set_id"])._length.nunique()
    if not lengths.eq(1).all():
        raise ValueError("Matching group mixes peptide lengths")
    hits = frame.loc[frame.hit == 1].set_index(["sample_id", "risk_set_id"])[affinity_column]
    targets = pandas.MultiIndex.from_frame(frame[["sample_id", "risk_set_id"]]).map(hits)
    if (numpy.abs(numpy.log10(values.to_numpy()) - numpy.log10(numpy.asarray(targets, dtype=float))) > caliper + 1e-10).any():
        raise ValueError("Saved negatives exceed their affinity caliper")
    if frame.duplicated(["sample_id", "risk_set_id", "peptide"]).any():
        raise ValueError("A matched risk set repeats a peptide")
    positives = pandas.MultiIndex.from_frame(frame.loc[frame.hit == 1, ["sample_id", "peptide"]])
    negatives = pandas.MultiIndex.from_frame(frame.loc[frame.hit == 0, ["sample_id", "peptide"]])
    if negatives.isin(positives).any():
        raise ValueError("An observed peptide is also labelled as a negative in its sample")


def sample_validation_mask(frame, fraction, seed):
    """Choose complete samples, independently of architecture-specific fit seeds."""
    if not 0 <= fraction < 1:
        raise ValueError("Validation fraction must be in [0, 1)")
    if fraction == 0:
        return numpy.zeros(len(frame), dtype=bool)
    samples = numpy.sort(frame.sample_id.unique())
    if len(samples) < 2:
        raise ValueError("Matched processing early stopping requires at least two training samples")
    count = min(len(samples) - 1, max(1, int(numpy.ceil(len(samples) * fraction))))
    selected = numpy.random.default_rng(seed).permutation(samples)[:count]
    return frame.sample_id.isin(selected).to_numpy()
