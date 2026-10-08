"""Model-bound execution and training evidence, without validation inference."""

from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import mhcgnomes
import pandas

from .common import normalize_allele_name, normalize_sequence_resolved_allele_name
from .training_provenance import file_sha256
from .version import __version__


BUNDLED_TRAINING_TABLES = {
    "affinity": "affinity_predictor_train_data.csv.bz2",
    "processing_with_flanks": "processing_predictor_with_flanks_train_data.csv.bz2",
    "processing_without_flanks": "processing_predictor_no_flank_train_data.csv.bz2",
}


def model_fingerprint(models_dir):
    """Hash every file and the sorted compact JSON filename-to-hash mapping."""
    directory = Path(models_dir)
    files = {path.relative_to(directory).as_posix(): file_sha256(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    if not files:
        raise ValueError("Model directory contains no files")
    digest = hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def generator_metadata(source_file):
    """Record code and dependency identities used to generate a report."""
    return {"mhcflurry": __version__, "python": platform.python_version(),
            "dependencies": {name: version(name) for name in
                             ("mhcgnomes", "numpy", "pandas", "torch")},
            "source_sha256": file_sha256(source_file),
            "source_files": {name: file_sha256(Path(__file__).parent / name) for name in (
                "allele_capabilities.py", "dla_evaluation.py", "common.py",
                "sample_disjointness.py", "training_provenance.py",
                "class1_affinity_predictor.py", "class1_processing_predictor.py",
                "class1_presentation_predictor.py")}}


def training_evidence(models_dir, alleles, lengths):
    """Count observations in bundled tables, preserving unknown host species.

    Counts include every row (including decoys) and are not a complete inventory
    of pretraining, training, development or selection. MHC species never
    substitutes for the species of the peptide source or experimental host.
    Missing tables/columns produce null counts, not zero evidence.
    """
    result = {}
    for component, filename in BUNDLED_TRAINING_TABLES.items():
        path = Path(models_dir) / filename
        if not path.exists():
            result[component] = {"path": filename, "status": "unavailable"}
            continue
        totals, strata = Counter(), Counter()
        total, unresolved_allele_rows = 0, 0
        for frame in pandas.read_csv(path, dtype=str, keep_default_na=False, chunksize=100_000):
            total += len(frame)
            allele_column = next((c for c in ("allele", "hla") if c in frame), None)
            if allele_column is None or "peptide" not in frame:
                unresolved_allele_rows += len(frame)
                continue
            mapping = {s: normalize_allele_name(s, raise_on_error=False)
                       for s in frame[allele_column].unique()}
            frame = frame.assign(canonical_allele=frame[allele_column].map(mapping))
            unresolved_allele_rows += int(frame.canonical_allele.isna().sum())
            frame = frame.loc[frame.canonical_allele.isin(alleles)].copy()
            frame["length"] = frame.peptide.str.len()
            for allele, count in frame.canonical_allele.value_counts().items():
                totals[allele] += int(count)
            for column in ("source_species", "host_species"):
                if column not in frame:
                    frame[column] = ""
            for keys, count in frame.groupby([
                    "canonical_allele", "length", "source_species", "host_species"
            ], dropna=False).size().items():
                strata[keys] += int(count)
        columns = pandas.read_csv(path, nrows=0).columns
        countable = "peptide" in columns and any(c in columns for c in ("allele", "hla"))
        result[component] = {
            "path": filename, "sha256": file_sha256(path), "status": "available",
            "total_rows": total, "source_provenance_present": "source_provenance" in columns,
            "unresolved_allele_rows": unresolved_allele_rows,
            "scope": "bundled table only; all labels; not complete model lineage",
            "allele_rows": {a: totals[a] if countable else None for a in alleles},
            "allele_length_rows": [
                {"allele": a, "length": n, "rows": sum(
                    count for (allele, length, _, _), count in strata.items()
                    if allele == a and length == n) if countable else None}
                for a in alleles for n in lengths],
            "species_strata": [
                {"allele": a, "length": int(n), "source_species": source or None,
                 "host_species": host or None, "rows": count}
                for (a, n, source, host), count in sorted(strata.items())],
        }
    return result


def capability_rows(predictor, alleles, lengths):
    """Describe no-flank modality support per requested allele and length.

    Support is determined from the loaded model configuration. This function
    does not run predictions or establish empirical accuracy. Processing is
    allele-independent and remains available for an unsupported allele when
    its length is supported. Presentation requires that exact resolved allele.
    """
    affinity = predictor.affinity_predictor
    processing = predictor.processing_predictor_without_flanks
    minimum, maximum = affinity.supported_peptide_lengths
    rows = []
    for raw in alleles:
        try:
            canonical = normalize_sequence_resolved_allele_name(raw)
        except ValueError:
            canonical = None
        key = affinity.canonicalize_allele_name(raw, raise_on_error=False) if canonical else None
        supported = key in affinity.supported_alleles
        key = key if supported else None
        parsed = mhcgnomes.parse(canonical, raise_on_error=False) if canonical else None
        sequence = (affinity.allele_to_sequence or {}).get(key)
        calibrated_key = affinity.percent_rank_calibrated_allele(key) if key else None
        for length in lengths:
            binding = supported and minimum <= length <= maximum
            processing_ok = processing is not None and 1 <= length <= processing.sequence_lengths["peptide"]
            presentation = bool(binding and processing_ok and predictor.weights_dataframe is not None
                                and "without_flanks" in predictor.weights_dataframe.index)
            rows.append({
                "requested_allele": raw, "canonical_allele": canonical,
                "model_allele": key, "mhc_species": parsed.species.name if parsed else None,
                "length": length, "context": "without_flanks",
                "model_input_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest() if sequence else None,
                "model_input_sequence_length": len(sequence) if sequence else None,
                "support_basis": "model configuration; not an execution test",
                "supported": {"binding": bool(binding), "processing": bool(processing_ok),
                              "presentation": presentation},
                "allele_status": "supported" if supported else "invalid" if canonical is None else "unsupported",
                "affinity_length_supported": bool(minimum <= length <= maximum),
                "percentile_calibration": {
                    "binding": {"available": bool(binding and calibrated_key), "model_allele": calibrated_key,
                                "reference_population": None},
                    "processing": {"available": bool(processing_ok and processing.percent_rank_transform is not None),
                                   "reference_population": None},
                    "presentation": {"available": bool(presentation and predictor.percent_rank_transform is not None),
                                     "reference_population": None},
                },
                "empirical_validation": {name: {"status": "unknown", "provenance": None}
                                         for name in ("binding", "processing", "presentation")},
            })
    return rows


def capability_report(models_dir, alleles, lengths=range(8, 16)):
    """Load an exact presentation bundle and report capability plus evidence.

    Parameters
    ----------
    models_dir : str or pathlib.Path
        Frozen presentation bundle directory. All files are hashed.
    alleles : iterable of str
        Requested allele names. Order and aliases are retained in the report.
    lengths : iterable of int
        Positive peptide lengths to inspect, including unsupported lengths.

    Returns
    -------
    dict
        JSON-serializable evidence. Null validation/calibration provenance means
        unknown; presence of a transform or training row does not establish it.
    """
    from .class1_presentation_predictor import Class1PresentationPredictor

    alleles, lengths = list(alleles), list(lengths)
    if not alleles or not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("Supply nonempty alleles and positive integer lengths")
    fingerprint = model_fingerprint(models_dir)
    predictor = Class1PresentationPredictor.load(str(models_dir))
    rows = capability_rows(predictor, alleles, lengths)
    canonical = sorted({r["canonical_allele"] for r in rows if r["canonical_allele"]})
    training = training_evidence(models_dir, canonical, lengths)
    if model_fingerprint(models_dir) != fingerprint:
        raise ValueError("Model bundle changed during report generation")
    return {
        "schema_version": 1, "generator": generator_metadata(__file__),
        "configuration": {"alleles": alleles, "lengths": lengths, "context": "without_flanks",
                          "random_seed": None},
        "model": fingerprint, "capabilities": rows, "training_evidence": training,
        "limitations": [
            "Support describes configuration; empirical validation remains unknown.",
            "Model input sequences may be pseudosequences, not full-length proteins.",
            "Bundled tables do not establish complete lineage or experimental host species.",
            "Calibration transforms do not identify their reference population or validate canine calibration.",
        ],
    }


def run_argv(argv, prog="mhcflurry eval allele-capabilities"):
    """Write a model-bound allele capability report as JSON."""
    import argparse

    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--alleles", nargs="+", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=list(range(8, 16)))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = capability_report(args.models_dir, args.alleles, args.lengths)
    with open(args.out, "x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
