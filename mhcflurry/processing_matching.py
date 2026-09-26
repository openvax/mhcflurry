"""Shared, auditable affinity/length matching for processing data."""

import hashlib
import json
from pathlib import Path

import numpy
import pandas

from .experiment_archive import sha256_file
from .common import DEFAULT_RANDOM_SEED, derive_seed, positive_int_arg, positive_float_arg


MATCHING_POLICY = "sample-length-affinity-random-without-replacement-v2"


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
                matching_implementation=MATCHING_POLICY)
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
            max_distance=args.max_affinity_distance,
            random_seed=getattr(args, "random_seed", DEFAULT_RANDOM_SEED))
    except ValueError as error:
        (directory / (name + ".failure.json")).write_text(json.dumps(
            {"sample_id": sample, "error": str(error)}, indent=2) + "\n")
        raise
    diagnostics["sample_id"] = sample
    (directory / (name + ".matching.json")).write_text(json.dumps(diagnostics, indent=2) + "\n")
    return matched


def _random_unique_assignments(targets, proteins, values, negative_proteins,
                               count, caliper, protein_caliper, rng):
    """Random sequential bipartite matching with augmenting-path repair.

    ``values`` must be sorted. Each hit has ``count`` slots; a negative can
    own only one slot. Random draws prefer eligible same-protein negatives.
    Repair can move earlier assignments to complete a feasible matching.
    No dense hit-by-negative matrix or recursive graph traversal is needed.
    """
    lower = numpy.searchsorted(values, numpy.nextafter(targets - caliper, -numpy.inf), side="left")
    upper = numpy.searchsorted(values, numpy.nextafter(targets + caliper, numpy.inf), side="right")
    # Equal-width calipers form ordered intervals. Earliest-target /
    # earliest-negative assignment gives an exact feasibility check in
    # linear time after sorting, before random augmenting paths. In sparse
    # strong-binder tails this avoids repeatedly exploring the same exhausted
    # candidate component for hundreds of unmatched hits.
    feasible = numpy.full((len(targets), count), -1, dtype="int64")
    cursor = 0
    for hit in numpy.argsort(targets, kind="stable"):
        cursor = max(cursor, int(lower[hit]))
        found = 0
        while cursor < upper[hit] and found < count:
            distance = float(values[cursor]) - float(targets[hit])
            if distance > caliper:
                break
            if abs(distance) <= caliper:
                feasible[hit, found] = cursor
                found += 1
            cursor += 1
    if (feasible < 0).any():
        return feasible
    owner = numpy.full(len(values), -1, dtype="int64")
    assigned = numpy.full(len(targets) * count, -1, dtype="int64")

    def eligible(slot, free_only=False):
        hit = slot // count
        candidates = numpy.arange(lower[hit], upper[hit])
        # Search bounds can round at an inclusive caliper boundary. Check
        # the actual subtraction too, as the saved-assignment validator does.
        candidates = candidates[numpy.abs(values[candidates] - targets[hit]) <= caliper]
        if free_only:
            candidates = candidates[owner[candidates] < 0]
        preferred = ((negative_proteins[candidates] == proteins[hit]) &
                     (numpy.abs(values[candidates] - targets[hit]) <= protein_caliper))
        if pandas.isna(proteins[hit]):
            preferred[:] = False
        return candidates, preferred

    def repair(root):
        # Every occupied negative leads to its owning slot. Parent edges
        # recover the alternating path when we reach a free negative.
        parents = {root: None}
        visited = numpy.zeros(len(assigned), dtype=bool)
        visited[root] = True
        queue = [root]
        for slot in queue:
            candidates, preferred = eligible(slot)
            order = numpy.concatenate((rng.permutation(candidates[preferred]),
                                       rng.permutation(candidates[~preferred])))
            # Each occupied negative has a distinct owning slot. Once that
            # slot is queued, revisiting its edge cannot extend the search.
            # Filter after permutation to preserve RNG consumption and the
            # exact original path order for every seed.
            owners = owner[order]
            order = order[(owners < 0) | ~visited[owners]]
            for negative in order:
                other = int(owner[negative])
                if other < 0:
                    while True:
                        assigned[slot] = negative
                        owner[negative] = slot
                        parent = parents[slot]
                        if parent is None:
                            return True
                        slot, negative = parent
                elif other not in parents:
                    parents[other] = (slot, int(negative))
                    visited[other] = True
                    queue.append(other)
        return False

    for slot in rng.permutation(len(assigned)):
        candidates, preferred = eligible(int(slot), free_only=True)
        if len(candidates):
            choices = candidates[preferred] if preferred.any() else candidates
            negative = int(rng.choice(choices))
            assigned[slot] = negative
            owner[negative] = slot
        else:
            repair(int(slot))
    return assigned.reshape(len(targets), count)


def _sample_unique_negatives(frame, negatives, hits, count, caliper,
                             protein_caliper, random_seed):
    """Match independently per sample/length, retaining the input hit order."""
    selected = numpy.full((len(hits), count), -1, dtype="int64")
    keys = ["sample_id", "peptide_len"]
    pools = {key: group.sort_values("log10_affinity", kind="stable")
             for key, group in negatives.groupby(keys, sort=False)}
    for (sample, length), group in hits.groupby(keys, sort=False):
        pool = pools.get((sample, length))
        if pool is None:
            continue
        # Separate calls / worker scheduling cannot change a sample's RNG.
        seed = derive_seed(random_seed, MATCHING_POLICY,
                           hashlib.sha256(str(sample).encode()).hexdigest(), int(length))
        positions = _random_unique_assignments(
            group.log10_affinity.to_numpy(dtype="float64"),
            group.protein_accession.astype(object).where(group.protein_accession.notna(), None).to_numpy(),
            pool.log10_affinity.to_numpy(dtype="float64"),
            pool.protein_accession.astype(object).where(pool.protein_accession.notna(), None).to_numpy(), count,
            caliper, protein_caliper, numpy.random.default_rng(seed))
        valid = positions >= 0
        rows = numpy.full(positions.shape, -1, dtype="int64")
        rows[valid] = pool.index.to_numpy()[positions[valid]]
        # Keep same-protein rows first for the existing match-rank convention.
        negative_proteins = pool.protein_accession.astype(object).where(
            pool.protein_accession.notna(), None).to_numpy()
        hit_proteins = group.protein_accession.astype(object).where(
            group.protein_accession.notna(), None).to_numpy()
        same = (valid & group.protein_accession.notna().to_numpy()[:, None] &
                (negative_proteins[positions.clip(0)] == hit_proteins[:, None]) &
                (numpy.abs(pool.log10_affinity.to_numpy()[positions.clip(0)] -
                           group.log10_affinity.to_numpy()[:, None]) <= protein_caliper))
        order = numpy.argsort(~same, axis=1, kind="stable")
        selected[group.index.to_numpy()] = numpy.take_along_axis(rows, order, axis=1)
    return selected


def make_affinity_controlled_risk_sets(
        frame, decoys_per_hit=10, same_protein_caliper=0.25,
        max_distance=0.25, random_seed=DEFAULT_RANDOM_SEED):
    """Return seeded random risk sets without reusing negatives per sample.

    Draw uniformly among available negatives within the affinity caliper,
    preferring the same source protein. Randomized assignment order and
    augmenting-path repair avoid false failures caused by greedy competition.
    This does not sample complete matchings uniformly. ``random_seed`` is
    recorded in diagnostics; identical ordered inputs and seed reproduce the
    assignments independently of sample batching.
    """
    if decoys_per_hit < 1 or int(decoys_per_hit) != decoys_per_hit:
        raise ValueError("decoys_per_hit must be positive")
    if not numpy.isfinite(max_distance) or not 0 <= max_distance <= 0.25:
        raise ValueError("Matching requires a finite affinity caliper between 0 and 0.25")
    if isinstance(random_seed, bool) or not isinstance(random_seed, (int, numpy.integer)):
        raise ValueError("Matching requires an integer random seed")
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
    hit_indices = frame.index[frame.hit == 1].to_numpy()
    hits = frame.loc[hit_indices].reset_index(drop=True)
    targets = hits.log10_affinity.to_numpy(dtype="float64")
    protein_distance = (min(max_distance, same_protein_caliper)
                        if same_protein_caliper is not None else max_distance)
    selected = _sample_unique_negatives(frame, negative_frame, hits,
        decoys_per_hit, max_distance, protein_distance, random_seed)
    counts = (selected >= 0).sum(axis=1)
    unresolved = numpy.flatnonzero(counts != decoys_per_hit)
    if len(unresolved):
        failures = [{"sample_id": str(hits.at[i, "sample_id"]), "peptide": hits.at[i, "peptide"],
                     "available": int(counts[i]), "log10_affinity": float(targets[i]),
                     "peptide_length": int(hits.at[i, "peptide_len"])} for i in unresolved]
        raise IncompleteProcessingMatches(
            "%d hits could not be assigned %d affinity/length-matched decoys "
            "without replacement within %.3g log10 units. Expand the scored candidate pool; "
            "no hits were silently dropped and no unmatched fallback was used. Examples: %s" % (
                len(unresolved), decoys_per_hit, max_distance, failures[:5]), failures)
    row_indices = numpy.column_stack([hit_indices, selected]).ravel()
    risk_ids = numpy.repeat(numpy.arange(len(hits)), decoys_per_hit + 1)
    match_ranks = numpy.tile(numpy.arange(decoys_per_hit + 1), len(hits))
    same_protein = numpy.ones((len(hits), decoys_per_hit + 1), dtype=bool)
    proteins = frame.protein_accession.astype(object).where(
        frame.protein_accession.notna(), None).to_numpy()
    hit_proteins = proteins[hit_indices]
    for i in range(decoys_per_hit):
        same_protein[:, i + 1] = (
            hits.protein_accession.notna().to_numpy() &
            (proteins[selected[:, i]] == hit_proteins) &
            (numpy.abs(frame.log10_affinity.to_numpy()[selected[:, i]] - targets) <= protein_distance))
    same_protein = same_protein.ravel()
    distances = numpy.abs(frame.log10_affinity.to_numpy()[row_indices] - targets[risk_ids])
    result = frame.loc[row_indices].copy().reset_index().rename(
        columns={"index": "source_row"})
    result["risk_set_id"] = numpy.asarray(risk_ids, dtype="int64")
    result["match_rank"] = numpy.asarray(match_ranks, dtype="int16")
    result["same_protein_match"] = numpy.asarray(same_protein, dtype=bool)
    result["log10_affinity_distance"] = numpy.asarray(
        distances, dtype="float64")
    negative_counts = result.loc[result.hit.eq(0)].groupby(["sample_id", "peptide"]).size()
    diagnostics = {
        "policy": MATCHING_POLICY,
        "max_log10_affinity_distance": max_distance,
        "risk_sets": int(result.risk_set_id.nunique()),
        "rows": int(len(result)),
        "decoys_per_hit": int(decoys_per_hit),
        "same_protein_caliper": same_protein_caliper,
        "fallback_decoys": int((~same_protein).sum()),
        "random_seed": int(random_seed),
        "replacement": False,
        "unique_negative_sample_peptides": int(len(negative_counts)),
        "max_negative_reuse": int(negative_counts.max()),
        "sampling": "uniform-eligible-same-protein-preferred-augmenting-paths",
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


def matched_training_data(frame, reference, decoys_per_hit=1, max_distance=0.25,
                          random_seed=DEFAULT_RANDOM_SEED):
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
        frame, decoys_per_hit=decoys_per_hit, max_distance=max_distance, random_seed=random_seed)
    matched["processing_matching_policy"] = MATCHING_POLICY
    matched["matching_affinity_reference_sha256"] = reference["sha256"]
    matched["matching_max_log10_distance"] = max_distance
    matched["matching_decoys_per_hit"] = decoys_per_hit
    matched["matching_random_seed"] = int(random_seed)
    return matched, diagnostics


def validate_matched_training_data(frame, policy="matched"):
    """Reject unmatched/corrupt cached tables before training or resuming."""
    if policy == "legacy":
        return
    if policy != "matched":
        raise ValueError("Unknown processing data policy: " + policy)
    required = ["sample_id", "peptide", "hit", "risk_set_id", "affinity_prediction",
                "processing_matching_policy", "matching_affinity_reference_sha256",
                "matching_max_log10_distance", "matching_decoys_per_hit", "matching_random_seed"]
    if frame.empty or set(required) - set(frame):
        raise ValueError("Processing requires matched training data. Regenerate it; "
                         "use --processing-data-policy legacy only for explicit historical replay.")
    if frame[required].isna().any().any() or not frame.processing_matching_policy.eq(MATCHING_POLICY).all():
        raise ValueError("Invalid processing matching metadata")
    for name in ("matching_affinity_reference_sha256", "matching_max_log10_distance", "matching_decoys_per_hit", "matching_random_seed"):
        if frame[name].nunique() != 1:
            raise ValueError("Inconsistent matching configuration: " + name)
    seed = frame.matching_random_seed.iloc[0]
    if isinstance(seed, (bool, numpy.bool_)) or not isinstance(seed, (int, numpy.integer)):
        raise ValueError("Invalid matching random seed")
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
    if negatives.duplicated().any():
        raise ValueError("Matching without replacement repeats a negative peptide within a sample")
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
