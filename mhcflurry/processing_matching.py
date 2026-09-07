"""Shared, auditable affinity/length matching for processing data."""

import hashlib
import json
from pathlib import Path

import numpy
import pandas

from .experiment_archive import sha256_file
from .common import positive_int_arg, positive_float_arg


MATCHING_POLICY = "sample-length-affinity-v1"


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
    if directory.exists() or Path(args.out).exists():
        raise ValueError("Use fresh processing data output; preserve prior matching artifacts")
    directory.mkdir(parents=True)
    sources = [args.hits, args.proteome_peptides or args.proteome_reference_csv]
    if getattr(args, "exclude_samples_file", None):
        sources.append(args.exclude_samples_file)
    provenance = {"policy": MATCHING_POLICY, "affinity_reference": reference,
                  "decoys_per_hit": args.decoys_per_hit,
                  "max_log10_affinity_distance": args.max_affinity_distance,
                  "candidate_pool_multiplier": args.ppv_multiplier,
                  "seed": getattr(args, "random_seed", None),
                  "inputs": {str(Path(p).resolve()): sha256_file(p) for p in sources}}
    (directory / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return reference


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
    global_pools = {}
    protein_pools = {}
    # One sequence per sample is one decoy, even if it maps to many proteins.
    positive_keys = pandas.MultiIndex.from_frame(
        frame.loc[frame.hit == 1, ["sample_id", "peptide"]])
    negative_frame = frame.loc[negatives].drop_duplicates(["sample_id", "peptide"])
    if pandas.MultiIndex.from_frame(negative_frame[["sample_id", "peptide"]]).isin(positive_keys).any():
        raise ValueError("An observed peptide is also labelled as a negative in its sample")
    for key, group in negative_frame.groupby(
            ["sample_id", "peptide_len"], sort=False):
        global_pools[key] = _sorted_pool(frame, group.index)
    for key, group in negative_frame.groupby(
            ["sample_id", "peptide_len", "protein_accession"],
            sort=False, dropna=False):
        protein_pools[key] = _sorted_pool(frame, group.index)

    row_indices = []
    risk_ids = []
    match_ranks = []
    same_protein = []
    distances = []
    fallback_count = 0
    incomplete = 0
    failures = []
    for risk_id, (hit_index, hit) in enumerate(
            frame.loc[frame.hit == 1].iterrows()):
        target = float(hit.log10_affinity)
        protein_key = (
            hit.sample_id, hit.peptide_len, hit.protein_accession)
        selected = _nearest(
            protein_pools.get(protein_key, (numpy.array([], dtype="int64"),
                                            numpy.array([], dtype="float64"))),
            target,
            decoys_per_hit,
            max_distance=min(max_distance, same_protein_caliper) if same_protein_caliper is not None else max_distance,
        )
        selected_same_protein = [True] * len(selected)
        if len(selected) < decoys_per_hit:
            needed = decoys_per_hit - len(selected)
            fallback = _nearest(
                global_pools.get((hit.sample_id, hit.peptide_len), (numpy.array([], dtype="int64"), numpy.array([]))),
                target,
                needed,
                excluded=selected,
                max_distance=max_distance,
            )
            fallback_count += len(fallback)
            selected.extend(fallback)
            selected_same_protein.extend([False] * len(fallback))
        if len(selected) != decoys_per_hit:
            incomplete += 1
            if len(failures) < 5:
                failures.append({"sample_id": str(hit.sample_id), "peptide": hit.peptide,
                                 "available": len(selected), "log10_affinity": target})
            continue

        row_indices.append(int(hit_index))
        risk_ids.append(risk_id)
        match_ranks.append(0)
        same_protein.append(True)
        distances.append(0.0)
        for rank, (index, is_same) in enumerate(
                zip(selected, selected_same_protein), 1):
            row_indices.append(index)
            risk_ids.append(risk_id)
            match_ranks.append(rank)
            same_protein.append(is_same)
            distances.append(abs(
                float(frame.at[index, "log10_affinity"]) - target))

    if incomplete:
        raise ValueError(
            "%d hits could not be assigned %d affinity/length-matched decoys "
            "within %.3g log10 units. Expand the scored candidate pool; "
            "no hits were silently dropped and no unmatched fallback was used. Examples: %s" % (
                incomplete, decoys_per_hit, max_distance, failures))
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
