"""Reproduce the exploratory DLA audit of Kaabinejadian et al. (PMID 42199926)."""

from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import numpy
import pandas

from .allele_capabilities import (
    BUNDLED_TRAINING_TABLES, capability_report, generator_metadata, model_fingerprint,
)
from .amino_acid import COMMON_AMINO_ACIDS
from .common import normalize_sequence_resolved_allele_name
from .sample_disjointness import audit_samples
from .training_provenance import file_sha256


STUDY = "pmid:42199926"
SOURCE_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S2589004226013507-"
# SHA-256 of the original workbooks, also checked against the MD5 values in
# PMC13200046's full-text XML. Only the original observation sheets are used.
WORKBOOK_SHA256 = {
    "mmc2.xlsx": "8a80e7eda8749cb60a79b78baf7d43e02f4edab25db600132a1aa3d073b73e1b",
    "mmc3.xlsx": "a87ff80df3ee7179b0bdc86c5fc49b00ad2c89cb40ca73133b97dc746096c25b",
    "mmc4.xlsx": "32eda8030256186a2c89df7bd58836d9e65ff5f060e1f1763673f92fed1f4052",
    "mmc5.xlsx": "c76b3712e2840f153812b4c3c3f84ec40040e224d7caf829a8563c677f0727ec",
    "mmc6.xlsx": "62248335784c8958333b2ce917e13a901e9b731577a78a13452b6349185d763a",
    "mmc7.xlsx": "2992966f4b9e1901cfc11bd6c75ed3bc2439d69dea529491434c2e3cd0cfdb30",
}
# Table 1 describes genotype, not experimentally assigned peptide restrictions.
TUMORS = {
    "mmc3.xlsx": ("Lola", "H58A", ["DLA-88*501:01"]),
    "mmc4.xlsx": ("Lola", "BB7.6", ["DLA-88*501:01"]),
    "mmc5.xlsx": ("163828A", "BB7.6", ["DLA-88*004:02"]),
    "mmc6.xlsx": ("Lily", "H58A", ["DLA-88*004:02", "DLA-88*501:01", "DLA-88*501:02"]),
    "mmc7.xlsx": ("Bogey", "BB7.6", ["DLA-88*003:02", "DLA-88*017:01", "DLA-88*006:01"]),
}


def prepare_observations(supplements_dir):
    """Read checksum-pinned original sheets; retain source rows and all lengths.

    Requires the optional ``openpyxl`` Excel reader. No workbook is modified.
    Lola's two antibody captures retain one biological sample identity. All
    HCT116 transductants share their parental biological sample identity.
    """
    records = []
    for filename, expected in WORKBOOK_SHA256.items():
        path = Path(supplements_dir) / filename
        if file_sha256(path) != expected:
            raise ValueError(f"Supplement checksum mismatch: {filename}")
        sheets = ([f"DLA-88-{code}" for code in ("00302", "01201", "50101")]
                  if filename == "mmc2.xlsx" else ["All Peptides"])
        for sheet in sheets:
            frame = pandas.read_excel(path, sheet_name=sheet, header=1, engine="openpyxl")
            frame = frame.rename(columns={"Peptides": "Peptide"})
            for offset, row in frame.iterrows():
                peptide = row.Peptide
                if not isinstance(peptide, str) or not peptide or len(peptide) != row.Length:
                    raise ValueError(f"Invalid peptide/length at {filename}:{sheet}:{offset + 3}")
                if filename == "mmc2.xlsx":
                    code = sheet.removeprefix("DLA-88-")
                    assigned = normalize_sequence_resolved_allele_name(f"DLA-88*{code[:3]}:{code[3:]}")
                    sample, antibody, genotype = "HCT116", "BB7.6", [assigned]
                    arm, host = "human_host_monoallelic", "Homo sapiens"
                else:
                    sample, antibody, genotype = TUMORS[filename]
                    assigned, arm, host = "", "canine_tumor", "Canis lupus familiaris"
                records.append({
                    "study_id": STUDY, "sample_id": sample, "arm": arm,
                    "capture_id": f"{filename}:{sheet}", "antibody": antibody,
                    "peptide": peptide, "length": len(peptide), "hit": 1,
                    "assigned_allele": assigned, "genotype": " ".join(genotype),
                    "mhc_species": "Canis lupus familiaris", "source_species": host,
                    "host_species": host, "source_accessions": row.Accession,
                    "source_file": filename, "source_sheet": sheet, "source_excel_row": offset + 3,
                })
    return pandas.DataFrame(records)


def peptide_overlap(cohort, source_paths):
    """Audit all source rows/alleles/labels against exact and I/L-folded strings.

    The returned row flags do not assert common study, donor or peptide–MHC
    identity. Absence from these tables cannot certify complete lineage.
    """
    flags = pandas.DataFrame(index=cohort.index)
    report = []
    for name, path in source_paths.items():
        peptides = set()
        for chunk in pandas.read_csv(path, usecols=["peptide"], dtype=str,
                                     keep_default_na=False, chunksize=100_000):
            peptides.update(chunk.peptide)
        folded = {p.replace("I", "L") for p in peptides}
        flags[f"{name}_exact_overlap"] = cohort.peptide.isin(peptides)
        flags[f"{name}_il_overlap"] = cohort.peptide.str.replace("I", "L").isin(folded)
        for arm, frame in cohort.groupby("arm", sort=True):
            unique = set(frame.peptide)
            report.append({
                "source": name, "source_sha256": file_sha256(path), "arm": arm,
                "unique_peptides": len(unique), "exact_overlap": len(unique & peptides),
                "il_overlap": sum(p.replace("I", "L") in folded for p in unique),
                "il_denominator": "distinct original peptide strings",
            })
    return flags, report


def observation_diagnostics(frame):
    """Report lengths and empirical 9-mer residue counts per capture.

    Residue counts are descriptive motifs, without alignment or a background
    enrichment claim. Tumor motifs remain unassigned to individual alleles.
    """
    result = []
    for capture, group in frame.groupby("capture_id", sort=True):
        unique = group.drop_duplicates("peptide")
        nine = unique.loc[unique.length.eq(9), "peptide"]
        result.append({
            "capture_id": capture, "arm": group.arm.iloc[0], "sample_id": group.sample_id.iloc[0],
            "assigned_allele": group.assigned_allele.iloc[0] or None,
            "observations": len(group), "unique_peptides": len(unique),
            "length_counts": {str(k): int(v) for k, v in unique.length.value_counts().sort_index().items()},
            "nine_mer_residue_counts": [dict(sorted(Counter(p[i] for p in nine).items())) for i in range(9)],
        })
    return result


def score_observations(predictor, frame):
    """Score fixed observations without inventing negatives or restrictions.

    Binding is the predicted minimum IC50 across a sample's stated genotype.
    The winning allele is explicitly a prediction. All genotype alleles must
    be supported; unsupported lengths/identities remain missing with reasons.
    Processing uses no flanks, because these observation sheets do not resolve
    protein mappings sufficiently to select unique source flanks.
    """
    scored = frame.copy()
    for column in ("affinity", "processing_score", "presentation_score"):
        scored[column] = numpy.nan
        scored[column + "_status"] = "unsupported_length"
    scored["predicted_best_allele"] = ""
    affinity = predictor.affinity_predictor
    processing = predictor.processing_predictor_without_flanks
    low, high = affinity.supported_peptide_lengths
    for _, group in frame.groupby("capture_id", sort=True):
        alleles = group.genotype.iloc[0].split()
        valid_sequence = group.peptide.map(lambda p: set(p) <= set(COMMON_AMINO_ACIDS))
        for column in ("affinity", "processing_score", "presentation_score"):
            scored.loc[group.index[~valid_sequence], column + "_status"] = "unsupported_residue"
        candidates = group.loc[valid_sequence]
        if processing is not None:
            proc = candidates.loc[candidates.length.between(1, processing.sequence_lengths["peptide"])]
            if len(proc):
                values = processing.predict(proc.peptide.tolist())
                if not numpy.isfinite(values).all():
                    raise ValueError("Nonfinite processing predictions")
                scored.loc[proc.index, "processing_score"] = values
                scored.loc[proc.index, "processing_score_status"] = "executed"
        else:
            scored.loc[candidates.index, "processing_score_status"] = "component_unavailable"
        keys = [affinity.canonicalize_allele_name(a, raise_on_error=False) for a in alleles]
        if not keys or any(key not in affinity.supported_alleles for key in keys):
            for column in ("affinity", "presentation_score"):
                scored.loc[candidates.index, column + "_status"] = "unsupported_genotype"
            continue
        binding = candidates.loc[candidates.length.between(low, high)]
        if binding.empty:
            continue
        matrix = numpy.asarray([affinity.predict(binding.peptide.tolist(), allele=a) for a in keys])
        if not numpy.isfinite(matrix).all():
            raise ValueError("Nonfinite affinity predictions")
        scored.loc[binding.index, "affinity"] = matrix.min(axis=0)
        scored.loc[binding.index, "affinity_status"] = "executed"
        scored.loc[binding.index, "predicted_best_allele"] = numpy.asarray(keys)[matrix.argmin(axis=0)]
        presentation = binding.loc[scored.loc[binding.index, "processing_score"].notna()]
        if (len(presentation) and predictor.weights_dataframe is not None
                and "without_flanks" in predictor.weights_dataframe.index):
            output = predictor.predict(presentation.peptide.tolist(), alleles=keys,
                                       include_affinity_percentile=False, verbose=0)
            values = output.presentation_score.to_numpy()
            if not numpy.isfinite(values).all():
                raise ValueError("Nonfinite presentation predictions")
            scored.loc[presentation.index, "presentation_score"] = values
            scored.loc[presentation.index, "presentation_score_status"] = "executed"
        else:
            scored.loc[presentation.index, "presentation_score_status"] = "component_unavailable"
    return scored


def score_summary(scored, seed=490, bootstrap_replicates=2000):
    """Summarize exploratory scores; bootstrap canine donors, never peptides.

    Lola antibody captures are deduplicated within donor. Confidence intervals
    describe the mean of donor medians, not accuracy or canine calibration.
    There is no independent biological replicate for each monoallelic arm.
    """
    summaries = []
    for capture, group in scored.groupby("capture_id", sort=True):
        row = {"capture_id": capture}
        for column in ("affinity", "processing_score", "presentation_score"):
            values = group[column].dropna()
            row[column] = {"executed_rows": len(values), "total_rows": len(group),
                           "median": float(values.median()) if len(values) else None,
                           "status_counts": {str(k): int(v) for k, v in group[column + "_status"].value_counts().items()},
                           "by_length": [
                               {"length": int(length), "total_rows": len(sub),
                                "executed_rows": int(sub[column].notna().sum()),
                                "median": float(sub[column].median()) if sub[column].notna().any() else None}
                               for length, sub in group.groupby("length", sort=True)]}
        summaries.append(row)
    canine = scored.loc[scored.arm.eq("canine_tumor")].drop_duplicates(["sample_id", "peptide"])
    bootstrap = {}
    for column in ("affinity", "processing_score", "presentation_score"):
        medians = canine.groupby("sample_id")[column].median().dropna()
        bootstrap[column] = {"donor_medians": medians.to_dict(), "mean_of_donor_medians": None,
                             "percentile_95_ci": None}
        if len(medians) >= 2:
            rng = numpy.random.default_rng(seed)
            replicates = rng.choice(medians.to_numpy(), size=(bootstrap_replicates, len(medians))).mean(axis=1)
            bootstrap[column].update(mean_of_donor_medians=float(medians.mean()),
                                     percentile_95_ci=numpy.quantile(replicates, [.025, .975]).tolist())
    return {"captures": summaries, "canine_donor_bootstrap": bootstrap,
            "seed": seed, "replicates": bootstrap_replicates,
            "estimand": "mean of within-donor medians on executable observed peptides; descriptive only"}


def evaluate_dla(models_dir, supplements_dir, out_dir):
    """Write a frozen, conservative historical-bundle canine evaluation.

    Parameters
    ----------
    models_dir : str or pathlib.Path
        Frozen presentation bundle to evaluate, without changing its weights.
    supplements_dir : str or pathlib.Path
        Original mmc2.xlsx through mmc7.xlsx from PMID 42199926.
    out_dir : str or pathlib.Path
        New output directory for observations, scores, evidence and sample audit.

    Returns
    -------
    dict
        Exploratory report. Historical bundled training tables alone never
        establish complete lineage; this recipe cannot certify a holdout.
    """
    from .class1_presentation_predictor import Class1PresentationPredictor

    directory, models_dir = Path(out_dir), Path(models_dir).resolve()
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")
    cohort = prepare_observations(supplements_dir)
    alleles = sorted({a for genotype in cohort.genotype.unique() for a in genotype.split()})
    capabilities = capability_report(models_dir, alleles)
    paths = {name: models_dir / filename for name, filename in BUNDLED_TRAINING_TABLES.items()
             if (models_dir / filename).exists()}
    flags, overlaps = peptide_overlap(cohort, paths)
    predictor = Class1PresentationPredictor.load(str(models_dir))
    scored = score_observations(predictor, cohort).join(flags)
    directory.mkdir(parents=True, exist_ok=False)
    cohort_path = directory / "observations.csv.gz"
    cohort.to_csv(cohort_path, index=False, compression={"method": "gzip", "mtime": 0})
    scored.to_csv(directory / "scores.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    # Unknown stages are omitted deliberately; [] means reviewed as unused in
    # the sample-audit contract. No biological sample IDs are invented for old
    # deduplicated training rows, even when their measurement_source is known.
    inventory = {
        "schema_version": 1,
        "identity_review": "Lola captures share one dog; all HCT116 transductants share a parental line. "
                           "Historical source/study/sample identities remain unresolved.",
        "models": [{"name": "evaluated-presentation-bundle", "kind": "presentation",
                    "artifacts": [str(models_dir / f) for f in capabilities["model"]["files"]],
                    "components": {
                        "affinity": {"complete": False, "evidence": "Bundled training snapshot only",
                                     "training": [{"path": str(p)} for n, p in paths.items() if n == "affinity"]},
                        "processing": {"complete": False, "evidence": "Bundled training snapshots only",
                                       "training": [{"path": str(p)} for n, p in paths.items() if n != "affinity"]},
                        "presentation": {"complete": False, "evidence": "Training/selection sources unavailable"},
                    }}],
    }
    (directory / "lineage.json").write_text(json.dumps(inventory, indent=2) + "\n")
    audit = audit_samples(directory / "lineage.json", cohort_path, directory / "sample_audit")
    if model_fingerprint(models_dir) != capabilities["model"]:
        raise ValueError("Model bundle changed during evaluation")
    report = {
        "schema_version": 1, "generator": generator_metadata(__file__),
        "configuration": {"context": "without_flanks", "training_rows": "all alleles and labels",
                          "negative_or_background_scheme": None,
                          "bootstrap_unit": "canine donor", "bootstrap_seed": 490,
                          "bootstrap_replicates": 2000},
        "study": STUDY, "doi": "10.1016/j.isci.2026.115975", "raw_ms": "PXD074485",
        "sources": {f: {"url": SOURCE_URL + f, "sha256": digest} for f, digest in WORKBOOK_SHA256.items()},
        "excel_reader": {"openpyxl": version("openpyxl")},
        "model_sha256": capabilities["model"]["sha256"],
        "observations_sha256": file_sha256(cohort_path), "scores_sha256": file_sha256(directory / "scores.csv.gz"),
        "sample_disjointness": {k: audit[k] for k in ("status", "models", "cohort")},
        "held_out_performance": {name: {"status": "not_evaluated", "metrics": None}
                                 for name in ("binding", "processing", "presentation", "calibration")},
        "peptide_overlap": overlaps, "diagnostics": observation_diagnostics(cohort),
        "exploratory_scores": score_summary(scored),
        "limitations": [
            "No sample is eligible for held-out claims with this incomplete historical inventory.",
            "Peptide overlap is scanned across all bundled training alleles and labels; it is not donor/study identity.",
            "No sampled background or measured negatives: AP, precision-at-N, AUROC and calibration are unevaluated.",
            "MS ligands are not quantitative binding measurements or isolated processing labels.",
            "Human-host monoallelic evidence does not validate endogenous canine processing.",
            "Tumor genotype minima and best alleles are predictions, not experimental restrictions.",
            "Donor-bootstrap score intervals describe four dogs, not prediction accuracy or population coverage.",
            "Recover every component's training/development/selection and teacher lineage, or retrain with study/donor/peptide exclusions.",
            "Use a new untouched confirmation cohort after development on these observations.",
        ],
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    capabilities["exploratory_evaluation"] = {
        "report": "report.json", "sha256": file_sha256(directory / "report.json"),
        "study": STUDY, "status": "exploratory_only", "held_out_eligible_samples": 0,
    }
    (directory / "capabilities.json").write_text(
        json.dumps(capabilities, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def run_argv(argv, prog="mhcflurry eval dla"):
    """Run the checksum-pinned 2026 DLA observation audit."""
    import argparse

    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--supplements-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    evaluate_dla(args.models_dir, args.supplements_dir, args.out_dir)
