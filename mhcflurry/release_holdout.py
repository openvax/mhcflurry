# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build and validate the frozen evaluation holdout for model releases."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas

from .common import (
    normalize_class1_genotype,
    normalize_sequence_resolved_allele_name,
)


BENCHMARK_FILES = {
    "monoallelic": "benchmark.monoallelic.train_excluded.csv.bz2",
    "multiallelic": "benchmark.multiallelic.train_excluded.csv.bz2",
}
AFFINITY_PMHCS_FILE = "affinity_pmhcs.csv"
AFFINITY_SAMPLES_FILE = "affinity_samples.csv"
PROCESSING_SAMPLES_FILE = "processing_samples.csv"
PRESENTATION_SAMPLES_FILE = "presentation_samples.csv"
POLICY_FILE = "policy.json"
VALIDATION_FILE = "validation.json"
PRESENTATION_HOLDOUT_PMIDS = ("31154438",)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _benchmark_rows(path, chunksize):
    for chunk in pandas.read_csv(
            path,
            usecols=["peptide", "sample_id", "hit", "hla"],
            chunksize=chunksize):
        yield chunk


def _canonical_allele_mapping(values):
    result = {}
    for raw in pandas.Series(values).dropna().astype(str).unique():
        try:
            result[raw] = normalize_sequence_resolved_allele_name(raw)
        except ValueError:
            result[raw] = None
    return result


def build_release_holdout(
        data_dir,
        training_data,
        mass_spec_data,
        out_dir,
        presentation_holdout_pmids=PRESENTATION_HOLDOUT_PMIDS,
        chunksize=500_000):
    """Derive release exclusions for one frozen source-study holdout.

    Affinity is evaluated on every monoallelic benchmark sample and on the
    selected multiallelic holdout. Every pMHC from current affinity training
    that appears in those benchmark rows (hit or decoy) is excluded. Processing
    and presentation are evaluated only on the named multiallelic source-study
    holdout; presentation excludes those samples from training in full.
    """
    data_dir = Path(data_dir).resolve()
    training_data = Path(training_data).resolve()
    mass_spec_data = Path(mass_spec_data).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not training_data.is_file():
        raise ValueError(f"Missing affinity training data: {training_data}")
    if not mass_spec_data.is_file():
        raise ValueError(f"Missing annotated MS data: {mass_spec_data}")

    training = pandas.read_csv(
        training_data, usecols=["allele", "peptide"])
    training_allele_map = _canonical_allele_mapping(training.allele)
    training["canonical_allele"] = training.allele.astype(str).map(
        training_allele_map)
    training = training.loc[training.canonical_allele.notna()]
    training_pmhcs = set(zip(
        training.canonical_allele, training.peptide.astype(str), strict=True))

    sample_metadata = pandas.read_csv(
        mass_spec_data,
        usecols=["sample_id", "pmid", "format", "mhc_class"],
    ).drop_duplicates("sample_id")
    sample_metadata["sample_id"] = sample_metadata.sample_id.astype(str)
    sample_metadata["pmid"] = sample_metadata.pmid.astype(str)
    holdout_pmids = tuple(str(value) for value in presentation_holdout_pmids)
    presentation_samples = set(sample_metadata.loc[
        sample_metadata.mhc_class.eq("I")
        & sample_metadata.format.eq("MULTIALLELIC")
        & sample_metadata.pmid.isin(holdout_pmids),
        "sample_id",
    ])
    if not presentation_samples:
        raise ValueError(
            f"No multiallelic samples found for holdout PMIDs {holdout_pmids}")

    affinity_pmhcs = set()
    affinity_samples = set()
    evaluation_row_counts = {}
    hit_counts = {}
    input_files = {}
    genotype_cache = {}

    for kind, filename in BENCHMARK_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise ValueError(f"Missing release benchmark: {path}")
        row_count = 0
        hit_count = 0
        benchmark_samples = set()
        for rows in _benchmark_rows(path, chunksize=chunksize):
            rows["sample_id"] = rows.sample_id.astype(str)
            if kind == "multiallelic":
                rows = rows.loc[
                    rows.sample_id.isin(presentation_samples)].copy()
            else:
                affinity_samples.update(rows.sample_id)
            if rows.empty:
                continue
            row_count += len(rows)
            hit_count += int(rows.hit.sum())
            benchmark_samples.update(rows.sample_id)
            rows["hla"] = rows.hla.astype(str)
            rows["peptide"] = rows.peptide.astype(str)
            for genotype, genotype_rows in rows.groupby("hla", sort=False):
                alleles = genotype_cache.get(genotype)
                if alleles is None:
                    alleles = normalize_class1_genotype(genotype)
                    genotype_cache[genotype] = alleles
                peptides = genotype_rows.peptide.unique()
                for allele in alleles:
                    affinity_pmhcs.update(
                        (allele, peptide)
                        for peptide in peptides
                        if (allele, peptide) in training_pmhcs)
        evaluation_row_counts[kind] = row_count
        hit_counts[kind] = hit_count
        input_files[kind] = {
            "path": str(path),
            "sha256": _sha256(path),
        }
        if kind == "multiallelic":
            missing = sorted(presentation_samples - benchmark_samples)
            if missing:
                raise ValueError(
                    "Holdout samples missing from multiallelic benchmark: "
                    f"{missing}")

    affinity_frame = pandas.DataFrame(
        sorted(affinity_pmhcs), columns=["allele", "peptide"])
    affinity_samples_frame = pandas.DataFrame({
        "sample_id": sorted(affinity_samples),
    })
    processing_frame = pandas.DataFrame({
        "sample_id": sorted(presentation_samples),
    })
    presentation_frame = pandas.DataFrame({
        "sample_id": sorted(presentation_samples),
    })
    affinity_frame.to_csv(out_dir / AFFINITY_PMHCS_FILE, index=False)
    affinity_samples_frame.to_csv(
        out_dir / AFFINITY_SAMPLES_FILE, index=False)
    processing_frame.to_csv(out_dir / PROCESSING_SAMPLES_FILE, index=False)
    presentation_frame.to_csv(out_dir / PRESENTATION_SAMPLES_FILE, index=False)

    holdout_files = {}
    for filename, frame in (
            (AFFINITY_PMHCS_FILE, affinity_frame),
            (AFFINITY_SAMPLES_FILE, affinity_samples_frame),
            (PROCESSING_SAMPLES_FILE, processing_frame),
            (PRESENTATION_SAMPLES_FILE, presentation_frame)):
        holdout_files[filename] = {
            "rows": len(frame),
            "sha256": _sha256(out_dir / filename),
        }

    policy = {
        "schema_version": 1,
        "policy": {
            "affinity": (
                "Evaluate all monoallelic samples and the named multiallelic "
                "source-study holdout. Exclude every current-training pMHC "
                "that occurs in those benchmark rows, including decoys; "
                "expand multiallelic genotypes to every listed allele."
            ),
            "processing": (
                "Evaluate only the named multiallelic source-study holdout; "
                "enforce zero training sample-id overlap."
            ),
            "presentation": (
                "Evaluate only the named multiallelic source-study holdout "
                "and exclude those whole samples from training."
            ),
        },
        "input_files": {
            **input_files,
            "affinity_training_data": {
                "path": str(training_data),
                "sha256": _sha256(training_data),
            },
            "mass_spec_data": {
                "path": str(mass_spec_data),
                "sha256": _sha256(mass_spec_data),
            },
        },
        "presentation_holdout_pmids": list(holdout_pmids),
        "evaluation_row_counts": evaluation_row_counts,
        "evaluation_hit_counts": hit_counts,
        "holdout_files": holdout_files,
        "affinity_pmhc_count": len(affinity_frame),
        "affinity_sample_count": len(affinity_samples_frame),
        "processing_sample_count": len(processing_frame),
        "presentation_sample_count": len(presentation_frame),
    }
    with open(out_dir / POLICY_FILE, "w") as fd:
        json.dump(policy, fd, indent=2, sort_keys=True)
        fd.write("\n")
    return policy


def load_excluded_samples(path):
    """Load a generated one-column sample exclusion manifest."""
    frame = pandas.read_csv(path)
    if list(frame.columns) != ["sample_id"]:
        raise ValueError(
            f"Expected one sample_id column in {path}; "
            f"got {list(frame.columns)}")
    return set(frame.sample_id.astype(str))


def exclude_samples(frame, manifest, label):
    """Remove rows belonging to samples in ``manifest`` and report counts."""
    excluded = load_excluded_samples(manifest)
    sample_ids = frame.sample_id.astype(str)
    mask = sample_ids.isin(excluded)
    print(
        f"Release holdout excluded {mask.sum()} rows across "
        f"{sample_ids.loc[mask].nunique()} {label} samples")
    return frame.loc[~mask].copy()


def exclude_affinity_pmhcs(frame, manifest):
    """Remove affinity rows whose canonical pMHC is in ``manifest``."""
    exclusions = pandas.read_csv(manifest)
    required = ["allele", "peptide"]
    if list(exclusions.columns) != required:
        raise ValueError(
            f"Expected columns {required} in {manifest}; "
            f"got {list(exclusions.columns)}")
    exclusion_index = pandas.MultiIndex.from_frame(exclusions[required])
    canonical = frame[required].copy()
    allele_map = _canonical_allele_mapping(canonical.allele)
    canonical["allele"] = canonical.allele.astype(str).map(allele_map)
    row_index = pandas.MultiIndex.from_frame(canonical)
    mask = row_index.isin(exclusion_index)
    print(
        f"Release holdout excluded {mask.sum()} affinity rows across "
        f"{row_index[mask].nunique()} unique pMHCs")
    return frame.loc[~mask].copy()


def validate_holdout_manifests(holdout_dir):
    """Validate the exclusion manifests against the frozen policy record."""
    holdout_dir = Path(holdout_dir).resolve()
    policy_path = holdout_dir / POLICY_FILE
    with open(policy_path) as fd:
        policy = json.load(fd)
    if policy.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported release holdout policy schema: {policy_path}")
    expected_files = {
        AFFINITY_PMHCS_FILE,
        AFFINITY_SAMPLES_FILE,
        PROCESSING_SAMPLES_FILE,
        PRESENTATION_SAMPLES_FILE,
    }
    holdout_files = policy.get("holdout_files", {})
    if set(holdout_files) != expected_files:
        raise ValueError(
            f"Release holdout policy has unexpected manifests: "
            f"{sorted(holdout_files)}")
    for filename, expected in holdout_files.items():
        path = holdout_dir / filename
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected.get("sha256"):
            raise ValueError(
                f"Release holdout manifest checksum mismatch: {path}")
        actual_rows = len(pandas.read_csv(path))
        if actual_rows != expected.get("rows"):
            raise ValueError(
                f"Release holdout manifest row count mismatch: {path}")
    return policy


def validate_release_holdout(
        holdout_dir,
        affinity_training_data,
        processing_training_data,
        presentation_training_data,
        out=None):
    """Fail if any frozen evaluation identity remains in release training."""
    holdout_dir = Path(holdout_dir).resolve()
    policy = validate_holdout_manifests(holdout_dir)
    affinity = pandas.read_csv(
        affinity_training_data, usecols=["allele", "peptide"])
    affinity_map = _canonical_allele_mapping(affinity.allele)
    affinity["allele"] = affinity.allele.astype(str).map(affinity_map)
    affinity = affinity.loc[affinity.allele.notna()]
    affinity_exclusions = pandas.read_csv(
        holdout_dir / AFFINITY_PMHCS_FILE)
    affinity_overlap = pandas.MultiIndex.from_frame(
        affinity[["allele", "peptide"]]).isin(
            pandas.MultiIndex.from_frame(affinity_exclusions)).sum()

    def count_sample_overlaps(training_data, manifest):
        excluded = load_excluded_samples(holdout_dir / manifest)
        overlap_rows = 0
        for chunk in pandas.read_csv(
                training_data,
                usecols=["sample_id"],
                dtype={"sample_id": str},
                chunksize=500_000):
            overlap_rows += chunk.sample_id.isin(excluded).sum()
        return int(overlap_rows)

    processing_overlap = count_sample_overlaps(
        processing_training_data, PROCESSING_SAMPLES_FILE)
    presentation_overlap = count_sample_overlaps(
        presentation_training_data, PRESENTATION_SAMPLES_FILE)

    result = {
        "schema_version": 1,
        "policy_sha256": _sha256(holdout_dir / POLICY_FILE),
        "holdout_files": policy["holdout_files"],
        "affinity_overlap_rows": int(affinity_overlap),
        "processing_overlap_rows": int(processing_overlap),
        "presentation_overlap_rows": int(presentation_overlap),
    }
    if out:
        with open(out, "w") as fd:
            json.dump(result, fd, indent=2, sort_keys=True)
            fd.write("\n")
    nonzero = {key: value for key, value in result.items()
               if key.endswith("_overlap_rows") and value}
    if nonzero:
        raise ValueError(f"Release holdout validation failed: {nonzero}")
    return result


def make_parser(prog="mhcflurry train release-holdout"):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--data-dir", required=True)
    build.add_argument("--training-data", required=True)
    build.add_argument("--mass-spec-data", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument(
        "--presentation-holdout-pmids",
        nargs="+",
        default=list(PRESENTATION_HOLDOUT_PMIDS),
    )
    build.add_argument("--chunksize", type=int, default=500_000)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--holdout-dir", required=True)
    validate.add_argument("--affinity-training-data", required=True)
    validate.add_argument("--processing-training-data", required=True)
    validate.add_argument("--presentation-training-data", required=True)
    validate.add_argument("--out")
    return parser


def run_argv(argv=None, prog="mhcflurry train release-holdout"):
    args = make_parser(prog=prog).parse_args(argv)
    if args.command == "build":
        build_release_holdout(
            data_dir=args.data_dir,
            training_data=args.training_data,
            mass_spec_data=args.mass_spec_data,
            out_dir=args.out_dir,
            presentation_holdout_pmids=args.presentation_holdout_pmids,
            chunksize=args.chunksize,
        )
    else:
        validate_release_holdout(
            holdout_dir=args.holdout_dir,
            affinity_training_data=args.affinity_training_data,
            processing_training_data=args.processing_training_data,
            presentation_training_data=args.presentation_training_data,
            out=args.out,
        )
    return 0


if __name__ == "__main__":
    run_argv()
