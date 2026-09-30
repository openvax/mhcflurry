"""Regression tests for complete contributor lineage and sample-level audits."""

import copy
import importlib.util
import json
from pathlib import Path

import pandas
import pytest

from mhcflurry import Class1AffinityPredictor
from mhcflurry.cli.reassign_mass_spec_training_data import reassign_mass_spec_training_data
from mhcflurry.release_holdout import load_excluded_samples, run_argv
from mhcflurry.sample_disjointness import SampleAliases, audit_samples, exclude_source_samples
from mhcflurry.training_provenance import (
    PROVENANCE_COLUMN, annotate_sources, decode_sources, deduplicate_measurements,
    encode_sources, file_sha256,
)


def source(study="1", sample="001", complete=True):
    return {"schema_version": 1, "complete": complete, "study_id": study,
            "sample_id": sample, "assay_id": "assay", "source": "fixture",
            "source_row": "0", "source_sha256": "a" * 64}


def write_csv(path, rows):
    pandas.DataFrame(rows).to_csv(path, index=False)
    return path


def test_annotation_dedup_preserves_measurements_and_every_contributor(tmp_path):
    path = write_csv(tmp_path / "raw.csv", [
        dict(peptide="SIINFEKL", value=10, study="1", sample="001"),
        dict(peptide="SIINFEKL", value=20, study="2", sample="002"),
        dict(peptide="GILGFVFTL", value=30, study="3", sample="003"),
    ])
    original = pandas.read_csv(path, dtype={"study": str, "sample": str})
    annotated = annotate_sources(original, path, "fixture", study_column="study", sample_column="sample")
    result = deduplicate_measurements(annotated, ["peptide"])
    pandas.testing.assert_frame_equal(result.drop(columns=PROVENANCE_COLUMN), original.drop_duplicates("peptide"))
    contributors = decode_sources(result.iloc[0][PROVENANCE_COLUMN])
    assert {(r["study_id"], r["sample_id"], r["source_row"]) for r in contributors} == {
        ("pmid:1", "001", "0"), ("pmid:2", "002", "1")}
    assert {r["source_sha256"] for r in contributors} == {file_sha256(path)}
    annotated.loc[1, PROVENANCE_COLUMN] = ""
    assert any(not r["complete"] for r in decode_sources(
        deduplicate_measurements(annotated, ["peptide"]).iloc[0][PROVENANCE_COLUMN]))


def test_iedb_curation_retains_assays_that_collapse_to_one_measurement(tmp_path):
    path = tmp_path / "iedb.csv"
    rows = [dict(Class="I", **{
        "Name.6": "HLA-A*02:01", "Name": "SIINFEKL", "Units": "nM",
        "Quantitative measurement": 20., "Measurement Inequality": "=",
        "Method": "purified MHC", "Qualitative Measurement": "Positive",
        "Authors": "An Author", "PubMed ID": study, "Assay IRI": assay,
        "sample_id": sample,
    }) for study, assay, sample in [("123", "assay:a", "001"), ("456", "assay:b", "002")]]
    path.write_text("IEDB grouped headers\n" + pandas.DataFrame(rows).to_csv(index=False))
    script = Path(__file__).resolve().parents[1] / "downloads-generation/data_curated/curate.py"
    spec = importlib.util.spec_from_file_location("curation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.load_data_iedb(path)
    assert len(result) == 1
    assert result.iloc[0].measurement_value == 20
    contributors = decode_sources(result.iloc[0][PROVENANCE_COLUMN])
    assert {(r["study_id"], r["sample_id"], r["assay_id"]) for r in contributors} == {
        ("pmid:123", "001", "assay:a"), ("pmid:456", "002", "assay:b")}


def test_whole_sample_exclusion_and_export_retains_provenance(tmp_path):
    rows = [dict(allele="HLA-A*02:01", peptide=peptide, measurement_kind="mass_spec",
                 measurement_inequality="<", measurement_value=100,
                 source_provenance=encode_sources(contributors))
            for peptide, contributors in [
                ("SIINFEKL", [source()]), ("GILGFVFTL", [source()]),
                ("AAAAAAAA", [source(sample="other"), source()]),
                ("LLLLLLLL", [source(sample="")]),
                ("VVVVVVVV", [source(study="2")])]]
    path = write_csv(tmp_path / "train.csv", rows)
    exclusions = write_csv(tmp_path / "exclude.csv", [dict(study_id="1", sample_id="001")])
    result = reassign_mass_spec_training_data(path, exclude_source_samples_manifest=exclusions,
                                             out_csv=tmp_path / "filtered.csv")
    assert result.peptide.tolist() == ["VVVVVVVV"]  # unrelated peptides from the sample removed too
    assert result.iloc[0][PROVENANCE_COLUMN] == rows[-1][PROVENANCE_COLUMN]
    predictor = Class1AffinityPredictor(metadata_dataframes={
        "train_data": result, "model_selection_data": result})
    predictor.save(str(tmp_path / "models"))
    for name in ("train_data", "model_selection_data"):
        saved = pandas.read_csv(tmp_path / "models" / (name + ".csv.bz2"))
        assert saved[PROVENANCE_COLUMN].tolist() == result[PROVENANCE_COLUMN].tolist()


def audit_fixture(tmp_path):
    train = write_csv(tmp_path / "train.csv", [dict(source_provenance=encode_sources([source("1", "training")]))])
    cohort = write_csv(tmp_path / "cohort.csv", [
        dict(row_id=i, study_id=study, sample_id=sample, hit=hit, peptide="SIINFEKL")
        for i, (study, sample, hit) in enumerate([
            ("1", "001", 1), ("1", "001", 0), ("2", "002", 1), ("2", "002", 0)])])
    artifact = tmp_path / "weights.npz"
    artifact.write_bytes(b"fixture model identity")
    component = dict(complete=True, evidence="Reviewed all source and selection tables", pretraining=[],
                     training=[dict(path=train.name)], development=[], selection=[])
    inventory = dict(schema_version=1, identity_review="Reviewed shared specimens across these fixture studies",
                     models=[dict(name="new", kind="presentation", artifacts=[artifact.name],
                                  components={name: copy.deepcopy(component) for name in
                                              ("affinity", "processing", "presentation")})])
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    return inventory, inventory_path, cohort, train


def test_audit_union_includes_selection_and_preserves_identical_positive_negative_rows(tmp_path):
    inventory, path, cohort, train = audit_fixture(tmp_path)
    selection = write_csv(tmp_path / "selection.csv", [dict(study_id="1", sample_id="001")])
    old = copy.deepcopy(inventory["models"][0])
    old["name"] = "old"
    old["components"]["affinity"]["selection"] = [dict(path=selection.name, identity_mode="sample_table")]
    inventory["models"].append(old)
    inventory["models"].append(dict(name="external", kind="external", components={}))
    path.write_text(json.dumps(inventory))
    report = audit_samples(path, cohort, tmp_path / "audit")
    assert report["status"] == "not_verified"
    assert report["external_status"] == "not_verified"
    kept = pandas.read_csv(tmp_path / "audit/cohort.csv.gz", dtype={"sample_id": str})
    assert kept.row_id.tolist() == [2, 3]
    assert kept.hit.tolist() == [1, 0]
    assert report["cohort"]["retained_samples"] == 1
    assert report["cohort"]["input_positives"] == 2
    assert report["input_files"][str(train)]["sha256"] == file_sha256(train)
    assert len(report["generator"]["source_sha256"]) == 64
    assert report["generator"]["random_seed"] is None
    assert report["generator"]["arguments"]["inventory_path"] == str(path)
    with pytest.raises(FileExistsError):
        audit_samples(path, cohort, tmp_path / "audit")


@pytest.mark.parametrize("missing", ["provenance", "component", "selection", "review", "coverage", "artifacts"])
def test_missing_evidence_never_certifies_disjointness(tmp_path, missing):
    inventory, path, cohort, train = audit_fixture(tmp_path)
    component = inventory["models"][0]["components"]["affinity"]
    if missing == "provenance":
        write_csv(train, [dict(allele="HLA-A*02:01", peptide="SIINFEKL")])
    elif missing == "component":
        del inventory["models"][0]["components"]["processing"]
    elif missing == "selection":
        del component["selection"]
    elif missing == "review":
        del inventory["identity_review"]
    elif missing == "coverage":
        component["complete"] = False
    else:
        inventory["models"][0]["artifacts"] = []
    path.write_text(json.dumps(inventory))
    report = audit_samples(path, cohort, tmp_path / "audit")
    assert report["status"] == "not_verified"
    assert report["cohort"]["retained_rows"] == 0


def test_missing_sample_excludes_entire_study_but_missing_study_is_unresolved(tmp_path):
    _, path, cohort, train = audit_fixture(tmp_path)
    write_csv(train, [dict(source_provenance=encode_sources([source("1", "")]))])
    report = audit_samples(path, cohort, tmp_path / "known-study")
    assert report["cohort"]["retained_rows"] == 2
    write_csv(train, [dict(source_provenance=encode_sources([source("", "sample")]))])
    report = audit_samples(path, cohort, tmp_path / "unknown-study")
    assert report["cohort"]["retained_rows"] == 0


def test_aliases_match_shared_specimens_and_reject_conflicts_and_cycles(tmp_path):
    _, path, cohort, train = audit_fixture(tmp_path)
    write_csv(train, [dict(source_provenance=encode_sources([source("3", "old-name")]))])
    aliases = write_csv(tmp_path / "aliases.csv", [dict(
        study_id="3", sample_id="old-name", canonical_study_id="1", canonical_sample_id="001")])
    report = audit_samples(path, cohort, tmp_path / "aliases", aliases_path=aliases)
    assert report["cohort"]["retained_rows"] == 2
    resolver = SampleAliases(aliases)
    assert resolver.resolve("pmid:3", "old-name") == ("pmid:1", "001")
    with aliases.open("a") as stream:
        stream.write("1,001,3,old-name\n")
    with pytest.raises(ValueError, match="Cyclic"):
        SampleAliases(aliases)
    with aliases.open("a") as stream:
        stream.write("1,001,4,another\n")
    with pytest.raises(ValueError, match="Conflicting"):
        SampleAliases(aliases)


def test_cli_gates_unresolved_reports_and_preserves_old_sample_ids(tmp_path):
    _, path, cohort, train = audit_fixture(tmp_path)
    command = ["audit-samples", "--inventory", str(path), "--cohort", str(cohort)]
    assert run_argv(command + ["--out-dir", str(tmp_path / "passed")]) == 0
    write_csv(train, [dict(peptide="SIINFEKL")])
    with pytest.raises(ValueError, match="not verified"):
        run_argv(command + ["--out-dir", str(tmp_path / "failed")])
    assert (tmp_path / "failed/sample_disjointness.json").is_file()
    assert run_argv(command + ["--out-dir", str(tmp_path / "report"), "--report-only"]) == 0
    sample_file = tmp_path / "samples.csv"
    sample_file.write_text("sample_id\n001\nNA\n")
    assert load_excluded_samples(sample_file) == {"001", "NA"}


def test_malformed_lineage_and_unknown_contributors_cannot_hide(tmp_path):
    _, path, cohort, train = audit_fixture(tmp_path)
    write_csv(train, [dict(source_provenance=encode_sources([source("3"), source(complete=False)]))])
    report = audit_samples(path, cohort, tmp_path / "incomplete")
    assert report["cohort"]["retained_rows"] == 0
    write_csv(train, [dict(source_provenance='{"sample_id":"001"}')])
    with pytest.raises(ValueError, match="source_provenance"):
        audit_samples(path, cohort, tmp_path / "malformed")
    assert not (tmp_path / "malformed").exists()


def test_empty_and_unknown_exclusions_do_not_change_measurements(tmp_path):
    frame = pandas.DataFrame({"peptide": ["SIINFEKL"]})
    manifest = tmp_path / "empty.csv"
    manifest.write_text("study_id,sample_id\n")
    pandas.testing.assert_frame_equal(exclude_source_samples(frame, manifest), frame)
    pandas.testing.assert_frame_equal(exclude_source_samples(frame.iloc[:0], manifest), frame.iloc[:0])


def test_cohort_metadata_conflicts_and_missing_studies(tmp_path):
    _, path, cohort, _ = audit_fixture(tmp_path)
    rows = pandas.read_csv(cohort, dtype=str).drop(columns="study_id")
    rows.to_csv(cohort, index=False)
    report = audit_samples(path, cohort, tmp_path / "unknown")
    assert report["cohort"]["retained_rows"] == 0
    metadata = write_csv(tmp_path / "studies.csv", [
        dict(sample_id="001", study_id="1"), dict(sample_id="002", study_id="2")])
    report = audit_samples(path, cohort, tmp_path / "resolved", sample_metadata=metadata)
    assert report["status"] == "disjoint"
    with metadata.open("a") as stream:
        stream.write("001,3\n")
    with pytest.raises(ValueError, match="Conflicting"):
        audit_samples(path, cohort, tmp_path / "conflict", sample_metadata=metadata)


def test_dedup_accepts_repeated_index_labels_without_changing_retained_indices():
    frame = pandas.DataFrame({"peptide": ["A", "B", "A"],
                              "source_provenance": [encode_sources([source(str(i))]) for i in range(3)]},
                             index=[5, 5, 8])
    result = deduplicate_measurements(frame, ["peptide"])
    assert result.index.tolist() == [5, 5]
    assert result.peptide.tolist() == ["A", "B"]
    assert len(decode_sources(result.iloc[0][PROVENANCE_COLUMN])) == 2


def test_incomplete_provenance_stays_unknown_after_deduplication(tmp_path):
    _, path, cohort, train = audit_fixture(tmp_path)
    frame = pandas.DataFrame({"peptide": ["A", "A"],
                              "source_provenance": [encode_sources([source("3")]), ""]})
    deduplicate_measurements(frame, ["peptide"]).to_csv(train, index=False)
    report = audit_samples(path, cohort, tmp_path / "unknown-contributor")
    assert report["cohort"]["retained_rows"] == 0


def test_unnamespaced_study_cannot_be_assumed_distinct(tmp_path):
    _, path, cohort, train = audit_fixture(tmp_path)
    write_csv(train, [dict(source_provenance=encode_sources([source("paper-name", "specimen")]))])
    report = audit_samples(path, cohort, tmp_path / "unqualified")
    assert report["cohort"]["retained_rows"] == 0
    aliases = write_csv(tmp_path / "aliases.csv", [dict(
        study_id="paper-name", sample_id="", canonical_study_id="3", canonical_sample_id="")])
    assert audit_samples(path, cohort, tmp_path / "resolved", aliases_path=aliases)["status"] == "disjoint"


@pytest.mark.parametrize("level", ["root", "model", "component", "source"])
def test_unknown_inventory_fields_cannot_hide_extra_training_sources(tmp_path, level):
    inventory, path, cohort, _ = audit_fixture(tmp_path)
    model = inventory["models"][0]
    component = model["components"]["affinity"]
    target = {"root": inventory, "model": model, "component": component,
              "source": component["training"][0]}[level]
    target["finetuning"] = [{"path": "overlapping-source.csv"}]
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="Unrecognized"):
        audit_samples(path, cohort, tmp_path / "invalid")


@pytest.mark.parametrize("study", ["pmid:nan", "pmid:0", "pmid:"])
def test_invalid_pubmed_identifiers_are_unresolved(tmp_path, study):
    _, path, cohort, train = audit_fixture(tmp_path)
    write_csv(train, [dict(source_provenance=encode_sources([source(study)]))])
    assert audit_samples(path, cohort, tmp_path / "invalid-pmid")["cohort"]["retained_rows"] == 0


def test_numeric_pubmed_formats_resolve_to_same_study(tmp_path):
    _, path, cohort, train = audit_fixture(tmp_path)
    write_csv(train, [dict(source_provenance=encode_sources([source("PMID:0001.0")]))])
    assert audit_samples(path, cohort, tmp_path / "normalized")["cohort"]["retained_rows"] == 2
