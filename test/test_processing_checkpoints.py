"""Best/terminal processing states stay independent and survive serialization."""

import json

import numpy
import pandas
import pytest
import torch

from mhcflurry import Class1ProcessingNeuralNetwork, Class1ProcessingPredictor
from mhcflurry.flanking_encoding import FlankingEncoding


@pytest.fixture
def retained_model(monkeypatch):
    import mhcflurry.class1_processing_neural_network as module
    # Make the best state unmistakably different from the terminal optimizer state.
    monkeypatch.setattr(module, "copy_module_state_dict_to_cpu", lambda model: {
        name: torch.zeros_like(value, device="cpu")
        for name, value in model.state_dict().items()})
    model = Class1ProcessingNeuralNetwork(
        max_epochs=2, early_stopping=False, validation_split=0.5,
        minibatch_size=2, convolutional_filters=4, convolutional_kernel_size=3,
        n_flank_length=2, c_flank_length=2, dropout_rate=0,
        save_all_checkpoints=True, restore_best_weights=True)
    model.fit(FlankingEncoding(["SIINFEKL"] * 4, ["AA"] * 4, ["CC"] * 4),
              [0, 1, 0, 1], seed=42, verbose=-1, progress_print_interval=None)
    return model


def test_retention_is_independent_and_has_epoch_steps(retained_model):
    model = retained_model
    assert set(model.checkpoint_weights) == {"best", "terminal"}
    assert all(not w.any() for w in model.checkpoint_weights["best"])
    assert any(w.any() for w in model.checkpoint_weights["terminal"])
    assert all(not w.any() for w in model.get_weights())
    before = [w.copy() for w in model.checkpoint_weights["terminal"]]
    with torch.no_grad():
        next(model.network().parameters()).add_(1)
    for expected, actual in zip(before, model.checkpoint_weights["terminal"]):
        numpy.testing.assert_array_equal(actual, expected)
    info = model.fit_info[-1]
    assert info["optimizer_steps"] == [1, 2]
    assert len(info["epoch_seconds"]) == len(info["loss"]) == 2
    assert info["stop_reason"] == "max_epochs"
    assert json.loads(json.dumps(model.get_config()))["checkpoint_weights"] == {}


def test_processing_checkpoint_save_load_and_refit_clear(tmp_path, retained_model):
    predictor = Class1ProcessingPredictor([retained_model])
    source = tmp_path / "source"
    predictor.save(str(source))
    loaded = Class1ProcessingPredictor.load(str(source))
    for policy in ("best", "terminal"):
        for expected, actual in zip(retained_model.checkpoint_weights[policy],
                                    loaded.models[0].checkpoint_weights[policy]):
            numpy.testing.assert_array_equal(actual, expected)
    # Refit the loaded model without checkpoint retention.
    model = loaded.models[0]
    model.hyperparameters.update(save_all_checkpoints=False, max_epochs=1)
    model.fit(FlankingEncoding(["SIINFEKL"] * 4, ["AA"] * 4, ["CC"] * 4),
              [0, 1, 0, 1], seed=43, verbose=-1, progress_print_interval=None)
    assert model.checkpoint_weights == {}
    destination = tmp_path / "refit"
    loaded.save(str(destination))
    manifest = pandas.read_csv(destination / "manifest.csv")
    assert manifest.checkpoint_best_weights.isna().all()
    assert manifest.checkpoint_terminal_weights.isna().all()
    assert Class1ProcessingPredictor.load(str(destination)).models[0].checkpoint_weights == {}
    # Saving in place must clear old references too, not expose stale states.
    loaded.save(str(source))
    assert Class1ProcessingPredictor.load(str(source)).models[0].checkpoint_weights == {}


def test_missing_checkpoint_fails_closed(tmp_path, retained_model):
    predictor = Class1ProcessingPredictor([retained_model])
    predictor.save(str(tmp_path))
    manifest = pandas.read_csv(tmp_path / "manifest.csv")
    (tmp_path / manifest.checkpoint_terminal_weights.iloc[0]).unlink()
    with pytest.raises(ValueError, match="Missing retained terminal"):
        Class1ProcessingPredictor.load(str(tmp_path))


def test_cpu_weight_snapshots_do_not_alias_parameters():
    model = Class1ProcessingNeuralNetwork(convolutional_filters=4)
    network = model.make_network(**model.network_hyperparameter_defaults.subselect(model.hyperparameters))
    before = network.get_weights_list()
    with torch.no_grad():
        next(network.parameters()).add_(1)
    assert not numpy.array_equal(before[0], network.get_weights_list()[0])
