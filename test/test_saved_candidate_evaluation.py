import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts/training"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("saved_candidate", directory / "evaluate_saved_candidate.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_inventory_requires_actual_weights(module, tmp_path):
    with (tmp_path / "manifest.csv").open("w") as fd:
        writer = csv.DictWriter(fd, fieldnames=["model_name"])
        writer.writeheader()
        writer.writerow({"model_name": "test"})
    with pytest.raises(ValueError, match="Missing"):
        module.model_inventory(tmp_path)
    (tmp_path / "weights_test.npz").write_bytes(b"weights")
    assert module.model_inventory(tmp_path)["models"] == 1


def test_copy_is_portable_and_rejects_changed_weights(module, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.npz").write_bytes(b"weights")
    destination = tmp_path / "new/nested/model"
    module.copy_predictor(source, destination)
    module.copy_predictor(source, destination)
    assert (destination / "weights.npz").read_bytes() == b"weights"
    (destination / "weights.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        module.copy_predictor(source, destination)


def test_comparison_always_has_frozen_holdout_and_predictions(module, tmp_path):
    args = SimpleNamespace(public_root=tmp_path / "public", release_holdout_dir=tmp_path / "holdout",
                           backend="gpu", gpus=1)
    command = module.comparison_command(args, tmp_path / "candidate", tmp_path / "eval", "presentation")
    assert command[command.index("--release-holdout-dir") + 1] == str(args.release_holdout_dir)
    assert "--skip-affinity-predictions" not in command
    assert "--limit-files" not in command
    assert command[:3] == ["mhcflurry", "eval", "compare-models"]


def test_output_cannot_overlap_source(module, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        module.main(["--candidate", str(tmp_path / "run"), "--out", str(tmp_path / "run/eval"),
                     "--public-root", str(tmp_path / "public"), "--release-holdout-dir", str(tmp_path / "holdout"),
                     "--source-commit", "test"])


def test_exact_phase_requires_saved_processing_run(module, tmp_path):
    with pytest.raises(ValueError, match="requires --exact-processing-run"):
        module.main(["--candidate", str(tmp_path / "run"), "--out", str(tmp_path / "eval"),
                     "--public-root", str(tmp_path / "public"), "--release-holdout-dir", str(tmp_path / "holdout"),
                     "--source-commit", "test", "--phase", "exact"])


def test_paired_comparison_adapter_preserves_both_sides(module):
    import pandas
    from paired_sample_metrics import comparison_long_frame

    data = pandas.DataFrame({"sample": ["one", "two"], "n": [100, 200], "n_pos": [10, 20],
                             "a_pr_auc": [0.3, 0.4], "b_pr_auc": [0.2, 0.3]})
    result = comparison_long_frame(data, units=["sample"], condition="model", metrics=["pr_auc"],
                                   labels=["candidate", "public"])
    assert result.pr_auc.tolist() == [0.3, 0.4, 0.2, 0.3]
    assert result.n.tolist() == [100, 200, 100, 200]
    assert result.model.tolist() == ["candidate", "candidate", "public", "public"]
