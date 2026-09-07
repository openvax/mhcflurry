import pandas
import pytest

from mhcflurry.training_folds import extract_training_folds


def test_reuse_preserves_original_fold_identity_and_index():
    data = pandas.DataFrame({"peptide": ["A", "B"], "fold_0": [False, True],
                             "fold_1": [True, False]}, index=[4, 9])
    clean, folds = extract_training_folds(data, 2, reuse=True)
    assert list(clean) == ["peptide"]
    pandas.testing.assert_frame_equal(folds, data[["fold_0", "fold_1"]])
    clean, folds = extract_training_folds(data, 2, reuse=False)
    assert list(clean) == ["peptide"]
    assert folds is None


@pytest.mark.parametrize("values", [[True, None], [True, "typo"], [True, True], [False, False]])
def test_reuse_rejects_invalid_or_empty_partitions(values):
    with pytest.raises(ValueError):
        extract_training_folds(pandas.DataFrame({"fold_0": values}), 1, reuse=True)


def test_reuse_requires_all_and_only_requested_folds():
    with pytest.raises(ValueError, match="exactly"):
        extract_training_folds(pandas.DataFrame({"fold_0": [True, False]}), 2, reuse=True)


def test_malformed_old_fold_columns_are_removed_or_rejected():
    data = pandas.DataFrame({"peptide": ["A", "B"], "fold_0_x": [True, False]})
    cleaned, folds = extract_training_folds(data, 1)
    assert list(cleaned) == ["peptide"]
    assert folds is None
    with pytest.raises(ValueError, match="exactly"):
        extract_training_folds(data, 1, reuse=True)


@pytest.mark.parametrize("policy", ["legacy", "matched"])
def test_processing_initialization_reuses_folds_without_suffixes(tmp_path, monkeypatch, policy):
    from mhcflurry.cli import train_processing_models_command as command

    data = pandas.DataFrame({
        "peptide": ["SIINFEKL", "GILGFVFTL"], "n_flank": ["AAAAA"] * 2,
        "c_flank": ["CCCCC"] * 2, "hit": [1, 0], "sample_id": ["a", "b"],
        "fold_0": [True, False], "fold_1": [False, True]})
    if policy == "matched":
        from mhcflurry.processing_matching import matched_training_data
        data = pandas.DataFrame({
            "peptide": ["SIINFEKL", "SIINFEKA", "GILGFVFTL", "GILGFVFTA"],
            "n_flank": ["AAAAA"] * 4, "c_flank": ["CCCCC"] * 4,
            "hit": [1, 0, 1, 0], "sample_id": ["a", "a", "b", "b"],
            "protein_accession": "p1", "affinity_prediction": [100, 110, 200, 210],
            "fold_0": [True, True, False, False], "fold_1": [False, False, True, True]})
        data, _ = matched_training_data(data, {"sha256": "a" * 64})
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)
    hp = tmp_path / "hp.yaml"
    hp.write_text("- max_epochs: 1\n  convolutional_filters: 4\n")
    out = tmp_path / "models"
    args = command.parser.parse_args([
        "--processing-data-policy", policy,
        "--data", str(data_path), "--hyperparameters", str(hp),
        "--out-models-dir", str(out), "--num-folds", "2", "--reuse-folds"])

    def unexpected_assignment(**kwargs):
        raise AssertionError("Existing folds must not be reassigned")

    monkeypatch.setattr(command, "assign_folds", unexpected_assignment)
    command.initialize_training(args)
    saved = pandas.read_csv(out / "train_data.csv.bz2")
    pandas.testing.assert_frame_equal(saved[["fold_0", "fold_1"]], data[["fold_0", "fold_1"]])
    assert not any(name.endswith(("_x", "_y")) for name in saved)
