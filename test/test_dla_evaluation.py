"""DLA evidence must never turn support, overlap filtering or scores into validation."""

import json
from types import SimpleNamespace

import numpy
import pandas
import pytest

from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork, Class1PresentationPredictor
from mhcflurry.allele_capabilities import capability_rows, model_fingerprint, training_evidence
from mhcflurry import dla_evaluation
from mhcflurry.dla_evaluation import (
    observation_diagnostics, peptide_overlap, score_observations, score_summary,
)


@pytest.fixture
def predictor():
    affinity = Class1AffinityPredictor(
        class1_pan_allele_models=[Class1NeuralNetwork()],
        allele_to_sequence={"DLA-88*001:01": "A" * 39, "DLA-88*003:02": "C" * 39,
                            "HLA-A*02:01": "D" * 39})
    processing = SimpleNamespace(sequence_lengths={"peptide": 15}, percent_rank_transform=None)
    return Class1PresentationPredictor(
        affinity_predictor=affinity, processing_predictor_without_flanks=processing,
        weights_dataframe=pandas.DataFrame(index=["without_flanks"]),
        percent_rank_transform=object())


def test_capabilities_preserve_dla_identity_and_unknown_evidence(predictor):
    rows = capability_rows(predictor, ["DLA-88*01:01", "DLA-88*001:01", "DLA-88*03:02"], [9])
    assert [r["model_allele"] for r in rows] == ["DLA-88*001:01", "DLA-88*001:01", "DLA-88*003:02"]
    assert rows[0]["model_input_sequence_sha256"] == rows[1]["model_input_sequence_sha256"]
    assert rows[0]["model_input_sequence_sha256"] != rows[2]["model_input_sequence_sha256"]
    assert all(all(r["supported"].values()) for r in rows)
    for row in rows:
        assert all(v["status"] == "unknown" for v in row["empirical_validation"].values())
        assert row["percentile_calibration"]["presentation"]["available"]
        assert row["percentile_calibration"]["presentation"]["reference_population"] is None
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("allele,status", [("DLA-88*999:99", "unsupported"),
                                         ("DLA-88", "invalid"), ("invalid", "invalid"),
                                         ("DLA-DRB1*001:01", "invalid")])
def test_no_substitution_for_unknown_alleles(predictor, allele, status):
    row = capability_rows(predictor, [allele], [9])[0]
    assert row["model_allele"] is None
    assert row["allele_status"] == status
    assert row["supported"] == {"binding": False, "presentation": False, "processing": True}
    assert row["model_input_sequence_sha256"] is None


def test_unsupported_lengths_do_not_claim_presentation(predictor):
    rows = capability_rows(predictor, ["DLA-88*001:01"], [7, 8, 15, 16])
    assert [r["supported"]["binding"] for r in rows] == [False, True, True, False]
    assert [r["supported"]["presentation"] for r in rows] == [False, True, True, False]
    predictor.processing_predictor_without_flanks = None
    row = capability_rows(predictor, ["DLA-88*001:01"], [9])[0]
    assert row["supported"] == {"binding": True, "processing": False, "presentation": False}


def test_fingerprint_changes_with_any_weight_or_metadata(tmp_path):
    (tmp_path / "weights.npz").write_bytes(b"weights1")
    first = model_fingerprint(tmp_path)
    (tmp_path / "weights.npz").write_bytes(b"weights2")
    second = model_fingerprint(tmp_path)
    assert first["sha256"] != second["sha256"]
    (tmp_path / "info.txt").write_text("metadata")
    assert model_fingerprint(tmp_path)["sha256"] != second["sha256"]


def test_training_counts_are_allele_length_specific_not_host_inference(tmp_path):
    pandas.DataFrame([
        dict(allele="DLA-88*01:01", peptide="SIINFEKL", hit=1),
        dict(allele="DLA-88*001:01", peptide="GILGFVFTL", hit=0),
        dict(allele="HLA-A*02:01", peptide="SIINFEKL", hit=1),
    ]).to_csv(tmp_path / "affinity_predictor_train_data.csv.bz2", index=False)
    result = training_evidence(tmp_path, ["DLA-88*001:01", "DLA-88*003:02"], [8, 9])
    affinity = result["affinity"]
    assert affinity["allele_rows"] == {"DLA-88*001:01": 2, "DLA-88*003:02": 0}
    assert [r["rows"] for r in affinity["allele_length_rows"]] == [1, 1, 0, 0]
    assert all(r["host_species"] is None and r["source_species"] is None for r in affinity["species_strata"])
    assert result["processing_without_flanks"]["status"] == "unavailable"
    assert not affinity["source_provenance_present"]


def test_missing_training_columns_are_unknown_and_explicit_species_survive(tmp_path):
    path = tmp_path / "affinity_predictor_train_data.csv.bz2"
    pandas.DataFrame([dict(peptide="SIINFEKL")]).to_csv(path, index=False)
    result = training_evidence(tmp_path, ["DLA-88*001:01"], [8])["affinity"]
    assert result["allele_rows"]["DLA-88*001:01"] is None
    assert result["allele_length_rows"][0]["rows"] is None
    pandas.DataFrame([dict(allele="DLA-88*001:01", peptide="SIINFEKL",
                           host_species="Homo sapiens", source_species="Homo sapiens")]).to_csv(path, index=False)
    result = training_evidence(tmp_path, ["DLA-88*001:01"], [8])["affinity"]
    assert result["species_strata"][0]["host_species"] == "Homo sapiens"


def test_overlap_scans_other_alleles_and_negative_rows(tmp_path):
    path = tmp_path / "training.csv"
    pandas.DataFrame([dict(peptide="SLLNFEKL", allele="HLA-A*02:01", hit=0),
                      dict(peptide="GILGFVFTL", allele="DLA-88*501:01", hit=1)]).to_csv(path, index=False)
    cohort = pandas.DataFrame(dict(peptide=["SIINFEKL", "GILGFVFTL", "AAAAAAAA"], arm="canine_tumor"))
    flags, reports = peptide_overlap(cohort, {"affinity": path})
    assert flags.affinity_exact_overlap.tolist() == [False, True, False]
    assert flags.affinity_il_overlap.tolist() == [True, True, False]
    assert reports[0]["exact_overlap"] == 1
    assert reports[0]["il_overlap"] == 2
    assert reports[0]["unique_peptides"] == 3


def observations():
    return pandas.DataFrame([
        dict(study_id="pmid:42199926", sample_id=sample, capture_id=capture,
             arm="canine_tumor", genotype=genotype, assigned_allele="", peptide=peptide,
             length=len(peptide), hit=1)
        for sample, capture, genotype, peptide in [
            ("Lola", "H58A", "DLA-88*001:01 DLA-88*003:02", "SIINFEKL"),
            ("Lola", "BB7.6", "DLA-88*001:01 DLA-88*003:02", "SIINFEKL"),
            ("Bogey", "other", "DLA-88*001:01 DLA-88*999:99", "SIINFEKL"),
            ("Lily", "long", "DLA-88*001:01", "A" * 16),
            ("Lily", "invalid", "DLA-88*001:01", "AAAAAAAA?"),
        ]])


def test_scoring_keeps_unknowns_and_predicted_restrictions_separate(predictor, monkeypatch):
    monkeypatch.setattr(predictor.affinity_predictor, "predict",
                        lambda peptides, allele: numpy.full(len(peptides), 20 if allele == "DLA-88*003:02" else 50))
    predictor.processing_predictor_without_flanks.predict = lambda peptides: numpy.full(len(peptides), .3)
    monkeypatch.setattr(predictor, "predict", lambda peptides, **kwargs:
                        pandas.DataFrame({"presentation_score": [.2] * len(peptides)}))
    original = observations()
    scored = score_observations(predictor, original)
    assert scored.affinity.iloc[:2].tolist() == [20, 20]
    assert scored.predicted_best_allele.iloc[:2].tolist() == ["DLA-88*003:02"] * 2
    assert scored.assigned_allele.eq("").all()
    assert scored.affinity.iloc[2:].isna().all()
    assert scored.affinity_status.iloc[2:].tolist() == ["unsupported_genotype", "unsupported_length", "unsupported_residue"]
    assert scored.processing_score.iloc[2] == .3  # allele-independent
    assert scored.presentation_score.iloc[2:].isna().all()
    pandas.testing.assert_frame_equal(original, observations())
    motifs = observation_diagnostics(scored)
    assert all(m["assigned_allele"] is None for m in motifs)


def test_bootstrap_uses_dogs_and_deduplicates_antibody_captures():
    frame = observations().iloc[:3].copy()
    for column in ("affinity", "processing_score", "presentation_score"):
        frame[column] = [1., 1., 3.]
        frame[column + "_status"] = "executed"
    first = score_summary(frame)
    duplicated = pandas.concat([frame, frame.iloc[:2]], ignore_index=True)
    second = score_summary(duplicated)
    assert first["canine_donor_bootstrap"] == second["canine_donor_bootstrap"]
    stat = first["canine_donor_bootstrap"]["affinity"]
    assert stat["donor_medians"] == {"Lola": 1., "Bogey": 3.}
    assert stat["mean_of_donor_medians"] == 2.
    assert stat["percentile_95_ci"] == [1., 3.]


@pytest.mark.parametrize("missing", ["processing", "presentation"])
def test_missing_components_are_not_reported_as_unsupported_lengths(predictor, monkeypatch, missing):
    monkeypatch.setattr(predictor.affinity_predictor, "predict",
                        lambda peptides, allele: numpy.full(len(peptides), 50.))
    predictor.processing_predictor_without_flanks.predict = lambda peptides: numpy.full(len(peptides), .3)
    if missing == "processing":
        predictor.processing_predictor_without_flanks = None
    else:
        predictor.weights_dataframe = None
    scored = score_observations(predictor, observations().iloc[:1])
    assert scored.affinity_status.iloc[0] == "executed"
    assert scored.presentation_score.isna().all()
    assert scored.presentation_score_status.iloc[0] == "component_unavailable"


def test_historical_recipe_never_exports_unresolved_samples(tmp_path, predictor, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "weights.npz").write_bytes(b"frozen model fixture")
    pandas.DataFrame([dict(allele="DLA-88*001:01", peptide="VVVVVVVV")]).to_csv(
        models / "affinity_predictor_train_data.csv.bz2", index=False)
    cohort = observations().iloc[:2].copy()
    cohort["source_species"] = "Canis lupus familiaris"
    monkeypatch.setattr(dla_evaluation, "prepare_observations", lambda _: cohort)
    monkeypatch.setattr(Class1PresentationPredictor, "load", lambda _: predictor)
    monkeypatch.setattr(dla_evaluation, "capability_report", lambda *_:
                        {"model": model_fingerprint(models)})
    scored = cohort.copy()
    for column in ("affinity", "processing_score", "presentation_score"):
        scored[column], scored[column + "_status"] = .5, "executed"
    monkeypatch.setattr(dla_evaluation, "score_observations", lambda *_: scored)
    # openpyxl is optional, and this test does not use workbooks.
    monkeypatch.setattr(dla_evaluation, "version", lambda _: "fixture")
    report = dla_evaluation.evaluate_dla(models, tmp_path, tmp_path / "out")
    assert report["sample_disjointness"]["status"] == "not_verified"
    assert report["sample_disjointness"]["cohort"]["retained_rows"] == 0
    assert all(v["metrics"] is None for v in report["held_out_performance"].values())
    assert all(r["exact_overlap"] == 0 for r in report["peptide_overlap"])
    inventory = json.loads((tmp_path / "out/lineage.json").read_text())
    for component in inventory["models"][0]["components"].values():
        assert component["complete"] is False
        assert "selection" not in component  # unavailable, not an empty unused stage


def test_supplement_checksum_checked_before_parsing(tmp_path):
    (tmp_path / "mmc2.xlsx").write_bytes(b"wrong file")
    with pytest.raises(ValueError, match="checksum mismatch"):
        dla_evaluation.prepare_observations(tmp_path)


def test_import_uses_only_original_sheets_and_preserves_biological_units(tmp_path, monkeypatch):
    calls = []

    def read_excel(path, sheet_name, header, engine):
        calls.append((path.name, sheet_name))
        assert header == 1
        return pandas.DataFrame({"Peptides" if path.name == "mmc5.xlsx" else "Peptide": ["SIINFEKL"],
                                 "Length": [8], "Accession": ["source-protein"]})

    monkeypatch.setattr(pandas, "read_excel", read_excel)
    monkeypatch.setattr(dla_evaluation, "file_sha256", lambda path:
                        dla_evaluation.WORKBOOK_SHA256[path.name])
    frame = dla_evaluation.prepare_observations(tmp_path)
    assert len(frame) == 8
    assert all(sheet == "All Peptides" for name, sheet in calls if name != "mmc2.xlsx")
    human = frame.loc[frame.arm.eq("human_host_monoallelic")]
    assert human.assigned_allele.nunique() == 3
    assert human.sample_id.unique().tolist() == ["HCT116"]
    assert human.host_species.eq("Homo sapiens").all()
    canine = frame.loc[frame.arm.eq("canine_tumor")]
    assert canine.sample_id.nunique() == 4
    assert canine.sample_id.eq("Lola").sum() == 2
    assert canine.assigned_allele.eq("").all()
    assert len(canine.loc[canine.sample_id.eq("Lily"), "genotype"].iloc[0].split()) == 3
    assert frame.source_excel_row.eq(3).all()


@pytest.mark.parametrize("command,module_name", [("dla", "dla_evaluation"),
                                                 ("allele-capabilities", "allele_capabilities")])
def test_eval_dispatch(command, module_name, monkeypatch):
    import importlib
    from mhcflurry.cli import eval_command

    module = importlib.import_module("mhcflurry." + module_name)
    monkeypatch.setattr(module, "run_argv", lambda argv, prog: (argv, prog))
    assert eval_command.run_argv([command, "--help"]) == (["--help"], "mhcflurry eval " + command)
    assert command in eval_command.format_help()
