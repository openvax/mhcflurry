"""Conservative biological-sample audits of explicitly inventoried lineage."""

import json
from pathlib import Path
import platform

import pandas

from . import training_provenance
from .training_provenance import (
    PROVENANCE_COLUMN, decode_sources, file_sha256, identity_text, study_identity,
)
from .version import __version__


STAGES = ("pretraining", "training", "development", "selection")
COMPONENTS = {
    "affinity": ("affinity",), "processing": ("processing",),
    "presentation": ("affinity", "processing", "presentation"),
    "external": ("external",),
}
IDENTITY_COLUMNS = ("study_id", "sample_id")


def _known_study(value):
    namespace, separator, identifier = value.partition(":")
    return bool(namespace and separator and identifier)


def read_identities(path):
    """Read literal identity strings, including leading zeroes and 'NA'."""
    return pandas.read_csv(path, dtype=str, keep_default_na=False)


class SampleAliases:
    """Resolve explicit specimen and study aliases; reject ambiguous cycles."""

    def __init__(self, path=None):
        self.mapping = {}
        if path is not None:
            frame = read_identities(path)
            columns = ["study_id", "sample_id", "canonical_study_id", "canonical_sample_id"]
            if list(frame.columns) != columns:
                raise ValueError(f"Aliases require columns {columns}")
            for row in frame.itertuples(index=False, name=None):
                source = (study_identity(row[0]), identity_text(row[1]))
                target = (study_identity(row[2]), identity_text(row[3]))
                if not source[0] or not target[0] or bool(source[1]) != bool(target[1]):
                    raise ValueError("Aliases must map studies to studies or samples to samples")
                if source in self.mapping and self.mapping[source] != target:
                    raise ValueError(f"Conflicting alias for {source}")
                self.mapping[source] = target
            for source in self.mapping:
                self.resolve(*source)

    def resolve(self, study, sample):
        """Return the canonical study/sample pair for one source identity."""
        current, seen = (study_identity(study), identity_text(sample)), set()
        while True:
            if current in seen:
                raise ValueError(f"Cyclic sample/study aliases: {current}")
            seen.add(current)
            target = self.mapping.get(current)
            if target is None:
                study_target = self.mapping.get((current[0], ""))
                target = (study_target[0], current[1]) if study_target else None
            if target is None or target == current:
                return current
            current = target


def source_identities(frame, aliases, *, identity_mode="source_provenance",
                      study_column="study_id", sample_column="sample_id"):
    """Yield contributor identities and provenance completeness for each row.

    Direct sample tables require an explicit inventory declaration. Assay IDs
    and source row numbers never substitute for biological samples.
    """
    if identity_mode not in ("source_provenance", "sample_table"):
        raise ValueError(f"Unknown identity_mode: {identity_mode}")
    for row in frame.to_dict("records"):
        if identity_mode == "sample_table":
            sources = [{"study_id": row.get(study_column, ""),
                        "sample_id": row.get(sample_column, ""), "complete": True}]
        else:
            sources = decode_sources(row.get(PROVENANCE_COLUMN, ""))
        pairs = [aliases.resolve(source["study_id"], source["sample_id"]) for source in sources]
        complete = bool(sources) and all(source["complete"] for source in sources)
        yield pairs, complete


def exclude_source_samples(frame, manifest, aliases=None):
    """Exclude whole held-out samples, or whole studies if samples are unknown.

    Any overlapping contributor removes the entire deduplicated measurement.
    Unknown lineage remains reported as unresolved. Missing study metadata on
    an exclusion conservatively matches that sample ID in every study.
    """
    aliases = SampleAliases(aliases)
    targets = read_identities(manifest)
    if list(targets.columns) != list(IDENTITY_COLUMNS):
        raise ValueError("Sample exclusions require study_id,sample_id columns")
    excluded = {aliases.resolve(*row) for row in targets.itertuples(index=False, name=None)}
    if any(not study and not sample for study, sample in excluded):
        raise ValueError("Sample exclusions require at least one identity")
    if any(study and not _known_study(study) for study, _ in excluded):
        raise ValueError("Study identities require an explicit namespace or a resolved alias")
    studies = {study for study, _ in excluded if study}
    unscoped = {sample for study, sample in excluded if not study}
    whole_studies = {study for study, sample in excluded if study and not sample}
    remove, unresolved = [], 0
    for pairs, complete in source_identities(frame, aliases):
        overlap = any(pair in excluded or pair[1] in unscoped or pair[0] in whole_studies
                      or (pair[0] in studies and not pair[1]) for pair in pairs)
        remove.append(overlap)
        unresolved += int(not overlap and (not complete or any(not _known_study(p[0]) for p in pairs)))
    print(f"Source holdout excluded {sum(remove)} rows; "
          f"{unresolved} retained rows have unresolved provenance")
    return frame.loc[~pandas.Series(remove, index=frame.index, dtype=bool)].copy()


def audit_samples(inventory_path, cohort_path, out_dir, *, aliases_path=None,
                  sample_metadata=None):
    """Write a new lineage audit and one shared sample-filtered cohort.

    Separation is conditional on the supplied inventory and identity review.
    Only samples disjoint from every inventoried MHCflurry model enter the
    exported cohort. External uncertainty is reported separately. Passing an
    audit does not make a previously inspected benchmark an untouched test set.
    """
    inventory_path, cohort_path, out_dir = map(Path, (inventory_path, cohort_path, out_dir))
    if out_dir.exists():
        raise FileExistsError(f"Audit output directory already exists: {out_dir}")
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("schema_version") != 1 or not inventory.get("models"):
        raise ValueError("Expected a schema_version=1 inventory with nonempty models")
    if set(inventory) - {"schema_version", "identity_review", "models"}:
        raise ValueError("Unrecognized inventory fields")
    aliases = SampleAliases(aliases_path)
    cohort = read_identities(cohort_path)
    if cohort.empty or "sample_id" not in cohort or cohort.sample_id.eq("").any():
        raise ValueError("Cohort requires rows with nonempty sample_id values")
    if "hit" in cohort:
        labels = pandas.to_numeric(cohort.hit, errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError("Cohort hit labels must be 0 or 1")
    if sample_metadata:
        metadata = read_identities(sample_metadata)
        if set(metadata.columns) != set(IDENTITY_COLUMNS):
            raise ValueError("Sample metadata requires study_id,sample_id columns")
        metadata = metadata.drop_duplicates()
        if metadata.sample_id.duplicated().any():
            raise ValueError("Conflicting study metadata for cohort sample")
        supplied = cohort.sample_id.map(metadata.set_index("sample_id").study_id).fillna("")
        if "study_id" in cohort:
            conflicts = (cohort.study_id.ne("") & supplied.ne("")
                         & cohort.study_id.map(study_identity).ne(supplied.map(study_identity)))
            if conflicts.any():
                raise ValueError("Cohort conflicts with supplied study metadata")
            cohort["study_id"] = cohort.study_id.where(cohort.study_id.ne(""), supplied)
        else:
            cohort["study_id"] = supplied
    if "study_id" not in cohort:
        cohort["study_id"] = ""
    identities = cohort[list(IDENTITY_COLUMNS)].drop_duplicates()
    canonical = [aliases.resolve(*row) for row in identities.itertuples(index=False, name=None)]
    file_records = {}

    def record_file(path):
        path = Path(path).resolve()
        if str(path) not in file_records:
            file_records[str(path)] = {"sha256": file_sha256(path)}
        return path

    def inventory_file(value):
        path = Path(value)
        return record_file(path if path.is_absolute() else inventory_path.parent / path)

    for path in (inventory_path, cohort_path, aliases_path, sample_metadata):
        if path:
            record_file(path)
    rows, summaries, names = [], [], set()
    identity_review = identity_text(inventory.get("identity_review", ""))
    for model in inventory["models"]:
        if set(model) - {"name", "kind", "artifacts", "components"}:
            raise ValueError("Unrecognized model inventory fields")
        name, kind = model.get("name"), model.get("kind")
        if not isinstance(name, str) or not name or name in names or kind not in COMPONENTS:
            raise ValueError("Models require unique names and a supported kind")
        names.add(name)
        reasons = []
        if not identity_review:
            reasons.append("No documented review of aliases and shared specimens")
        artifacts = model.get("artifacts", [])
        if not artifacts:
            reasons.append("No model artifacts inventoried")
        for artifact in artifacts:
            inventory_file(artifact)
        components = model.get("components", {})
        if set(components) != set(COMPONENTS[kind]):
            reasons.append("Missing or unexpected component inventory")
        samples, studies, unknown_studies = set(), set(), set()
        unknown_rows = 0
        for component in components.values():
            if set(component) - {"complete", "evidence", *STAGES}:
                raise ValueError("Unrecognized component inventory fields")
            if component.get("complete") is not True or not identity_text(component.get("evidence", "")):
                reasons.append("Component coverage is not declared complete with evidence")
            component_rows = 0
            for stage in STAGES:
                entries = component.get(stage)
                if not isinstance(entries, list):
                    reasons.append(f"Missing {stage} inventory (use [] only when reviewed as unused)")
                    continue
                for entry in entries:
                    if set(entry) - {"path", "identity_mode", "study_column", "sample_column"}:
                        raise ValueError("Unrecognized source entry fields")
                    path = inventory_file(entry["path"])
                    for chunk in pandas.read_csv(path, dtype=str, keep_default_na=False, chunksize=100_000):
                        component_rows += len(chunk)
                        for pairs, complete in source_identities(
                                chunk, aliases, identity_mode=entry.get("identity_mode", "source_provenance"),
                                study_column=entry.get("study_column", "study_id"),
                                sample_column=entry.get("sample_column", "sample_id")):
                            unknown_rows += int(not complete or any(not _known_study(pair[0]) for pair in pairs))
                            for study, sample in pairs:
                                if _known_study(study):
                                    studies.add(study)
                                    if sample:
                                        samples.add((study, sample))
                                    else:
                                        unknown_studies.add(study)
            if not component_rows:
                reasons.append("No source rows inventoried for a component")
        if unknown_rows:
            reasons.append(f"{unknown_rows} rows lack complete source/study provenance")
        for (study, sample), source in zip(canonical, identities.to_dict("records"), strict=True):
            if (study, sample) in samples or study in unknown_studies:
                status = "overlap"
                reason = ("Whole study excluded: training sample identities unavailable"
                          if study in unknown_studies else "Shared biological sample")
            elif reasons or not _known_study(study):
                status, reason = "unresolved", "; ".join(reasons) or "Cohort study identity unavailable"
            else:
                status, reason = "disjoint", "No shared sample in the reviewed lineage inventory"
            rows.append({**source, "model": name, "kind": kind, "status": status, "reason": reason})
        summaries.append({"name": name, "kind": kind, "unresolved_reasons": sorted(set(reasons)),
                          "known_samples": len(samples), "known_studies": len(studies),
                          "studies_without_sample_ids": sorted(unknown_studies)})
    table = pandas.DataFrame(rows)
    internal = table.loc[table.kind.ne("external")]
    if internal.empty:
        raise ValueError("Inventory must include at least one MHCflurry model")
    eligible = internal.groupby(list(IDENTITY_COLUMNS), dropna=False).status.apply(
        lambda values: values.eq("disjoint").all())
    kept = set(eligible.loc[eligible].index)
    mask = [tuple(row) in kept for row in cohort[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None)]
    filtered = cohort.loc[mask]
    out_dir.mkdir(parents=True, exist_ok=False)
    table.to_csv(out_dir / "sample_audit.csv", index=False)
    filtered.to_csv(out_dir / "cohort.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    external = table.loc[table.kind.eq("external")]
    report = {
        "schema_version": 1,
        "generator": {
            "package_version": __version__, "python_version": platform.python_version(),
            "pandas_version": pandas.__version__, "random_seed": None,
            "source_sha256": file_sha256(__file__),
            "training_provenance_sha256": file_sha256(training_provenance.__file__),
            "function": "mhcflurry.sample_disjointness.audit_samples",
            "arguments": {
                "inventory_path": str(inventory_path.resolve()),
                "cohort_path": str(cohort_path.resolve()), "out_dir": str(out_dir.resolve()),
                "aliases_path": str(Path(aliases_path).resolve()) if aliases_path else None,
                "sample_metadata": str(Path(sample_metadata).resolve()) if sample_metadata else None,
            },
        },
        "status": "disjoint" if internal.status.eq("disjoint").all() else "not_verified",
        "external_status": ("not_inventoried" if external.empty else
                            "disjoint" if external.status.eq("disjoint").all() else "not_verified"),
        "identity_review": identity_review, "models": summaries, "input_files": file_records,
        "cohort": {"input_rows": len(cohort), "retained_rows": len(filtered),
                   "input_samples": len(identities), "retained_samples": len(kept),
                   "sha256": file_sha256(out_dir / "cohort.csv.gz")},
        "sample_audit_sha256": file_sha256(out_dir / "sample_audit.csv"),
        "limitations": [
            "Separation is conditional on the supplied lineage and identity review.",
            "Peptide/MHC overlap must be checked independently.",
            "Every comparator must use the same exported cohort.",
            "An inspected benchmark does not become an untouched test set by passing an audit.",
        ],
    }
    for label, data in (("input", cohort), ("retained", filtered)):
        if "hit" in data:
            labels = pandas.to_numeric(data.hit)
            report["cohort"][f"{label}_positives"] = int(labels.eq(1).sum())
            report["cohort"][f"{label}_negatives"] = int(labels.eq(0).sum())
    (out_dir / "sample_disjointness.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
