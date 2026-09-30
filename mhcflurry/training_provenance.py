"""Preserve source identities without changing curated measurement weights."""

import hashlib
import json
from pathlib import Path

import pandas


PROVENANCE_COLUMN = "source_provenance"


def file_sha256(path):
    """Return the SHA-256 of an artifact's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_text(value):
    """Return a literal identity, preserving meaningful leading zeroes."""
    return "" if value is None or pandas.isna(value) else str(value).strip()


def study_identity(value):
    """Normalize numeric PubMed identifiers; retain explicit namespaces."""
    text = identity_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return "pmid:" + text if text.isdigit() else text


def encode_sources(records):
    """Serialize distinct source records in canonical order."""
    unique = {json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records}
    return "[" + ",".join(sorted(unique)) + "]"


def decode_sources(value):
    """Read source records, refusing malformed JSON schemas."""
    if value is None or (isinstance(value, float) and pandas.isna(value)) or value == "":
        return []
    try:
        records = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid source_provenance JSON") from error
    if not isinstance(records, list) or not all(
            isinstance(record, dict) and record.get("schema_version") == 1
            and isinstance(record.get("complete"), bool)
            and all(isinstance(record.get(key), str) for key in (
                "study_id", "sample_id", "assay_id", "source", "source_row", "source_sha256"))
            for record in records):
        raise ValueError("Invalid source_provenance records")
    for record in records:
        if record["complete"] and (
                not record["source_row"] or len(record["source_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in record["source_sha256"])):
            raise ValueError("Complete source_provenance requires a source row and SHA-256")
    return records


def annotate_sources(frame, path, source, *, study_column=None,
                     sample_column=None, assay_column=None):
    """Attach raw identities before filtering or deduplication.

    Missing study/sample IDs stay empty. File and row IDs support reconstruction
    but never substitute for biological samples. ``complete`` records retention
    of all source contributors, not availability of every sample identity.
    """
    result = frame.copy()
    digest = file_sha256(path)

    def values(column):
        return (frame[column].map(identity_text).tolist()
                if column and column in frame else [""] * len(frame))

    result[PROVENANCE_COLUMN] = [encode_sources([{
        "schema_version": 1, "complete": True, "source": source,
        "source_file": Path(path).name, "source_sha256": digest,
        "source_row": str(index), "study_id": study_identity(study),
        "sample_id": sample, "assay_id": assay,
    }]) for index, study, sample, assay in zip(
        frame.index, values(study_column), values(sample_column), values(assay_column), strict=True)]
    return result


def deduplicate_measurements(frame, subset):
    """Keep the historical first measurement and union all source contributors.

    Values, weighting and row order match ``drop_duplicates(subset)``. Unknown
    provenance on any contributor stays explicit beside known duplicates.
    """
    if PROVENANCE_COLUMN not in frame:
        return frame.drop_duplicates(subset).copy()
    # Position-based indexing also handles callers with duplicate index labels.
    original_index = frame.index
    frame = frame.reset_index(drop=True)
    result = frame.drop_duplicates(subset).copy()
    duplicated = frame.duplicated(subset, keep=False)
    for _, group in frame.loc[duplicated].groupby(subset, sort=False, dropna=False):
        records = []
        for value in group[PROVENANCE_COLUMN]:
            records.extend(decode_sources(value) or [{
                "schema_version": 1, "complete": False,
                "study_id": "", "sample_id": "", "assay_id": "",
                "source": "unknown", "source_row": "", "source_sha256": "",
            }])
        result.loc[group.index[0], PROVENANCE_COLUMN] = encode_sources(records)
    result.index = original_index.take(result.index)
    return result
