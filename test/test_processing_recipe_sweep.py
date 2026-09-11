"""The maintained processing factorial is paired, explicit and reproducible."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas
import pytest

from .test_processing_matching import training_frame


@pytest.fixture
def recipe_modules(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts/training"
    monkeypatch.syspath_prepend(str(directory))
    import generate_processing_recipe
    import run_exact_public_processing
    spec = importlib.util.spec_from_file_location("recipe_sweep", directory / "run_processing_kernel_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    return generate_processing_recipe, sweep, run_exact_public_processing


def test_recipe_factorial_has_all_joint_combinations(recipe_modules):
    generator, _, _ = recipe_modules
    records = generator.build_processing_recipe_conditions()
    assert len(records) == 32
    assert sum(axes["network_count"] for _, _, axes in records) == 64
    assert len({name for name, _, _ in records}) == 32
    combinations = set()
    for _, grid, axes in records:
        hp = grid[0]
        combinations.add((axes["family"], hp["optimizer"], hp["initialization_method"], hp["minibatch_size"]))
        assert hp["restore_best_weights"] and hp["save_all_checkpoints"]
        assert hp["convolutional_kernel_size"] == 11
        assert hp["convolutional_kernel_l1_l2"] == [0, 0]
        assert axes["fold_count"] == 2
    assert len(combinations) == 32


def test_ranking_confirmation_design(recipe_modules):
    generator, _, _ = recipe_modules
    records = generator.build_processing_confirmation_conditions()
    assert len(records) == 6
    assert sum(axes["network_count"] for _, _, axes in records) == 24
    assert {(hp[0]["optimizer"], axes["kernel_width"]) for _, hp, axes in records} == {
        (optimizer, width) for optimizer in ("adam", "rmsprop") for width in (11, 13, 15)}
    for _, grid, axes in records:
        hp = grid[0]
        assert hp["checkpoint_metric"] == "val_macro_ap"
        assert hp["monitor_validation_ranking"] and hp["restore_best_weights"] and hp["save_all_checkpoints"]
        assert hp["minibatch_size"] == 512 and hp["initialization_method"] == "none"
        assert hp["convolutional_kernel_l1_l2"] == [0, 0]
        assert axes["fold_count"] == 4


def test_confirmed_candidate_is_one_confirmation_condition(recipe_modules):
    generator, _, _ = recipe_modules
    named = generator.CONFIRMED_PROCESSING_CANDIDATE
    records = {name: (grid, axes) for name, grid, axes in generator.build_processing_confirmation_conditions()}
    grid, axes = records[named["condition"]]
    assert grid == [generator.confirmed_processing_candidate_hyperparameters()]
    assert (axes["kernel_width"], axes["optimizer"], axes["checkpoint_policy"]) == (
        named["kernel_width"], named["optimizer"], named["checkpoint_policy"])
    assert grid[0]["convolutional_kernel_size"] == 13 and grid[0]["optimizer_implementation"] == "pytorch"
    assert not named["release_accepted"]


def test_confirmed_candidate_yaml_emitter_matches_named_recipe(recipe_modules):
    import yaml
    generator, _, _ = recipe_modules
    script = Path(__file__).resolve().parents[1] / "scripts/training/generate_processing_recipe.py"
    result = subprocess.run([sys.executable, str(script), "--confirmed-candidate"],
                            capture_output=True, text=True, check=True)
    assert yaml.safe_load(result.stdout) == [generator.confirmed_processing_candidate_hyperparameters()]
    assert subprocess.run([sys.executable, str(script)], capture_output=True).returncode != 0


def test_final_230_candidate_v2_preset_uses_confirmed_short_flanks():
    repo = Path(__file__).resolve().parents[1]
    script = ("source scripts/training/release_recipes.sh; apply_final_230_candidate_v2_recipe; "
              'echo "$MHCFLURRY_RELEASE_RECIPE|$PROCESSING_SHORT_FLANKS_HYPERPARAMETERS|$PROCESSING_VARIANTS|'
              '$PROCESSING_MODES|$AFFINITY_MINIBATCH_SIZE|$PROCESSING_SHORT_FLANK_BOUNDARY_RADIUS"')
    result = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == ("final-2.3.0-candidate-v2|confirmed-ranking-candidate|"
                                     "with_flanks no_flank short_flanks|with_flanks,no_flank,short_flanks|1024|5")
    decision = json.loads((repo / "scripts/training/final_230_candidate_v2_recipe.json").read_text())
    v1 = json.loads((repo / "scripts/training/final_230_candidate_recipe.json").read_text())
    assert decision["name"] == "final-2.3.0-candidate-v2"
    assert decision["processing"]["legacy_architectures_per_variant"] == {
        "with_flanks": 128, "no_flank": 128, "short_flanks": 1}
    assert decision["processing"]["variants"] == v1["processing"]["variants"]
    assert decision["affinity"] == v1["affinity"] and decision["presentation"] == v1["presentation"]
    workflow = (repo / "scripts/training/pan_allele_release_full.sh").read_text()
    assert "final_230_candidate_v2_recipe.json" in workflow and "--confirmed-candidate" in workflow


def test_ranking_confirmation_tiny_workflow(tmp_path, recipe_modules, monkeypatch):
    generator, sweep, _ = recipe_modules
    records = generator.build_processing_confirmation_conditions()[:2]
    for _, grid, _ in records:
        grid[0].update(max_epochs=2, convolutional_filters=4, post_convolutional_dense_layer_sizes=[4])
    monkeypatch.setattr(generator, "build_processing_confirmation_conditions", lambda: records)
    frames = []
    for index in range(20):
        frame = training_frame().assign(sample_id="sample_%d" % index)
        frame["peptide"] = frame.peptide.str.replace("B", "R")
        for fold in range(4):
            frame["fold_%d" % fold] = (index + fold) % 4 != 0
        frames.append(frame)
    table = tmp_path / "frozen.csv.bz2"
    pandas.concat(frames, ignore_index=True).to_csv(table, index=False)
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pandas.DataFrame({"sample_id": ["unseen"]}).to_csv(holdout / "processing_samples.csv", index=False)
    out = tmp_path / "confirmation"
    args = ["--design", "ranking-confirmation", "--out", str(out), "--train-data", str(table),
            "--folds-from", str(table), "--public-root", str(tmp_path),
            "--release-holdout-dir", str(holdout), "--source-commit", "test",
            "--gpus", "0", "--num-jobs", "0", "--evaluation", "none"]
    assert sweep.main(args) == 0
    assert sweep.main(args) == 0
    assert (out / "validation_ranking_confirmation.pdf").stat().st_size > 1000
    for name, _, _ in records:
        predictions = pandas.read_csv(out / "checkpoint_predictions" / (name + ".validation_predictions.csv.gz"))
        assert set(predictions.checkpoint_policy) == {"best", "best_ap", "terminal"}
        assert set(predictions.fold_num) == {0, 1, 2, 3}
        configs = pandas.read_csv(out / name / "processing/models.unselected.short_flanks/manifest.csv").config_json
        for config in configs.map(json.loads):
            info = config["fit_info"][-1]
            fold = info["training_info"]["fold_num"]
            outer = set(predictions.loc[predictions.fold_num == fold, "sample_id"])
            assert not outer & set(info["ranking_validation_samples"])
            assert info["restored_checkpoint_policy"] == "best_ap"
            assert len(info["val_macro_ap"]) == 2
        metrics = pandas.read_csv(out / name / "checkpoint_per_sample.csv")
        assert set(metrics.checkpoint_policy) == {"best", "best_ap", "terminal"}


def test_recipe_sweep_real_tiny_fit_and_frozen_folds(tmp_path, recipe_modules, monkeypatch):
    generator, sweep, _ = recipe_modules
    records = generator.build_processing_recipe_conditions()[8:10]  # pre-LSUV, both families
    assert all(grid[0]["initialization_method"] == "lsuv_pre" for _, grid, _ in records)
    for _, grid, _ in records:
        grid[0].update(max_epochs=1, convolutional_filters=8,
                       cleavage_boundary_hidden_size=4, post_convolutional_dense_layer_sizes=[4])
    monkeypatch.setattr(generator, "build_processing_recipe_conditions", lambda: records)
    frames = []
    for index in range(20):
        frame = training_frame().assign(sample_id="sample_%d" % index)
        frame["peptide"] = frame.peptide.str.replace("B", "R")
        for fold in range(4):
            frame["fold_%d" % fold] = (index + fold) % 4 != 0
        frames.append(frame)
    table = tmp_path / "frozen.csv.bz2"
    pandas.concat(frames, ignore_index=True).to_csv(table, index=False)
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pandas.DataFrame({"sample_id": ["unseen"]}).to_csv(holdout / "processing_samples.csv", index=False)
    out = tmp_path / "experiment"
    args = ["--design", "training-recipe", "--out", str(out), "--train-data", str(table),
            "--folds-from", str(table), "--public-root", str(tmp_path),
            "--release-holdout-dir", str(holdout), "--source-commit", "test",
            "--gpus", "0", "--num-jobs", "0", "--evaluation", "none"]
    assert sweep.main(args) == 0
    assert sweep.main(args) == 0
    assert (out / "validation_training_recipe.pdf").stat().st_size > 1000
    assert not (out / "evaluation/cohort.json").exists()
    for name, _, _ in records:
        log = (out / "logs" / (name + "-train.log")).read_text()
        assert "Trained processing predictor with 2 networks" in log
        assert "Trained affinity predictor" not in log
        manifest = pandas.read_csv(out / name / "processing/models.selected.short_flanks/manifest.csv")
        assert len(manifest) == 2
        configs = [json.loads(value) for value in manifest.config_json]
        assert all(config["fit_info"][-1]["initialization"]["applied"] for config in configs)
        predictions = pandas.read_csv(out / "checkpoint_predictions" / (name + ".validation_predictions.csv.bz2"))
        assert set(predictions.checkpoint_policy) == {"best", "terminal"}
        assert set(predictions.fold_num) == {0, 1}
    corrupted = out / "checkpoint_predictions" / (records[0][0] + ".validation_predictions.csv.bz2")
    corrupted.write_bytes(corrupted.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Changed checkpoint prediction cache"):
        sweep.main(args)


def test_budget_deadline_stops_command_and_never_marks_complete(tmp_path, recipe_modules):
    _, _, driver_module = recipe_modules
    driver = driver_module.Driver(tmp_path, deadline=time.time() + 0.25)
    with pytest.raises(subprocess.CalledProcessError) as error:
        driver.run("bounded", [sys.executable, "-c", "import time; time.sleep(60)"])
    assert error.value.returncode == 124
    record = json.loads((tmp_path / "commands/bounded.json").read_text())
    assert record["budget_deadline_exceeded"]
    assert not (tmp_path / "stages/bounded.json").exists()
    driver = driver_module.Driver(tmp_path, deadline=time.time() - 1)
    with pytest.raises(TimeoutError, match="budget deadline"):
        driver.run("expired", [sys.executable, "-c", "raise AssertionError('must not launch')"])
