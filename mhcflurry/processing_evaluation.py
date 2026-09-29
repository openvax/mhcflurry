"""Integrity checks for a frozen, expanded processing evaluation cohort."""
import hashlib
import json
from pathlib import Path

import numpy
import pandas

from .experiment_archive import sha256_file
from .processing_matching import MATCHING_POLICY, validate_matched_training_data


IDENTITY = ["sample_id", "peptide", "protein_accession", "n_flank", "c_flank", "hla", "hit"]


def hit_identity(frame):
    """Ordered hit identities, including MHC alleles and protein context."""
    hits = frame.loc[frame.hit.eq(1), IDENTITY].fillna("").astype(str)
    return pandas.util.hash_pandas_object(hits, index=False).to_numpy()


def verify_hits(original, matched):
    """Fail if matching removes, duplicates, reorders or changes a held-out hit."""
    if not numpy.array_equal(hit_identity(original), hit_identity(matched)):
        raise ValueError("Expanded processing cohort changed held-out hits")


def load_processing_cohort(directory, original):
    """Read a checksummed 10:1 cohort and prove it retains the benchmark hits."""
    directory = Path(directory)
    metadata = json.loads((directory / "cohort.json").read_text())
    filename = metadata["cohort_file"]
    if Path(filename).name != filename:
        raise ValueError("Invalid processing cohort filename")
    path = directory / filename
    if sha256_file(path) != metadata["cohort_sha256"]:
        raise ValueError("Processing cohort checksum mismatch")
    if metadata["policy"] != MATCHING_POLICY or metadata["decoys_per_hit"] != 10:
        raise ValueError("Processing evaluation requires current-policy unique 10:1 negatives")
    frame = pandas.read_csv(path, dtype={"sample_id": str})
    frame[["n_flank", "c_flank"]] = frame[["n_flank", "c_flank"]].fillna("")
    validate_matched_training_data(frame)
    verify_hits(original, frame)
    if (set(frame.sample_id) != set(original.sample_id.astype(str)) or
            len(frame) != metadata["rows"] or int(frame.hit.sum()) != metadata["hits"] or
            frame.sample_id.nunique() != metadata["samples"] or
            not frame.matching_decoys_per_hit.eq(10).all() or
            not frame.matching_max_log10_distance.eq(metadata["max_log10_affinity_distance"]).all() or
            not frame.matching_random_seed.eq(metadata["seed"]).all() or
            not frame.matching_affinity_reference_sha256.eq(metadata["affinity_reference"]["sha256"]).all() or
            hashlib.sha256(hit_identity(frame).tobytes()).hexdigest() != metadata["hit_identity_sha256"]):
        raise ValueError("Processing cohort metadata disagrees with saved assignments")
    return frame, metadata
