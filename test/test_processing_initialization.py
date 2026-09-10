"""Processing LSUV uses real activation semantics and protects scalar heads."""

import numpy
import pytest
import torch

from mhcflurry import Class1ProcessingNeuralNetwork
from mhcflurry.flanking_encoding import FlankingEncoding
from mhcflurry.processing_initialization import (
    initialize_processing_network, processing_activation_variance,
    processing_hidden_layers)


def make_model(boundary=True):
    numpy.random.seed(42)
    torch.manual_seed(42)
    hp = dict(convolutional_filters=16, convolutional_kernel_size=5,
              n_flank_length=5, c_flank_length=5, convolutional_activation="relu",
              convolutional_padding_mode="unknown", post_convolutional_dense_layer_sizes=[8],
              dropout_rate=0.5)
    if boundary:
        hp.update(cleavage_boundary_flank_length=5, cleavage_boundary_peptide_length=5,
                  cleavage_boundary_hidden_size=8, cleavage_boundary_context_dropout=0.25)
    wrapper = Class1ProcessingNeuralNetwork(**hp)
    model = wrapper.make_network(**wrapper.network_hyperparameter_defaults.subselect(wrapper.hyperparameters))
    peptides = ["SIINFEKL", "GILGFVFTL", "NLVPMVATV", "AAAAAAAA"] * 8
    sequences = FlankingEncoding(peptides, ["ARNDC"] * 32, ["QEGHI"] * 32)
    inputs = {name: torch.as_tensor(value.copy()) for name, value in wrapper.network_input(sequences).items()}
    return model, inputs


@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("method", ["orthogonal", "lsuv_pre", "lsuv_post"])
def test_initialization_variance_and_protected_parameters(boundary, method):
    model, inputs = make_model(boundary)
    before = {name: value.clone() for name, value in model.named_parameters()}
    eligible = {name + ".weight" for name, _ in processing_hidden_layers(model)}
    report = initialize_processing_network(model, inputs, method)
    assert model.training
    for name, value in model.named_parameters():
        if name not in eligible:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert len(report["layers"]) == (5 if boundary else 3)
    if method.startswith("lsuv"):
        model.eval()
        for name, layer in processing_hidden_layers(model):
            variance = processing_activation_variance(model, layer, inputs, method == "lsuv_post")
            assert abs(variance - 1) <= 0.05, (name, variance)


def test_none_is_exact_noop_and_pre_post_are_different():
    model, inputs = make_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    initialize_processing_network(model, inputs, "none")
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    initialize_processing_network(model, inputs, "lsuv_pre")
    pre = model.conv1.weight.detach().clone()
    other, other_inputs = make_model()
    initialize_processing_network(other, other_inputs, "lsuv_post")
    assert not torch.allclose(pre, other.conv1.weight)


def test_failure_rolls_back_weights_and_removes_hooks(monkeypatch):
    import mhcflurry.processing_initialization as module
    model, inputs = make_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(module, "processing_activation_variance", lambda *a, **k: 0.0)
    with pytest.raises(ValueError, match="Degenerate"):
        initialize_processing_network(model, inputs, "lsuv_pre")
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    def fail(*args):
        raise RuntimeError("forward failed")
    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        processing_activation_variance(model, model.conv1, inputs)
    assert not model.conv1._forward_hooks


def test_fit_calibration_excludes_validation_rows_and_survives_save(tmp_path):
    from mhcflurry import Class1ProcessingPredictor
    model = Class1ProcessingNeuralNetwork(
        initialization_method="lsuv_pre", initialization_batch_size=3,
        convolutional_filters=8, convolutional_kernel_size=3,
        convolutional_activation="relu", n_flank_length=2, c_flank_length=2,
        max_epochs=1, minibatch_size=2, save_all_checkpoints=True, restore_best_weights=True)
    inputs = FlankingEncoding(["SIINFEKL", "GILGFVFTL", "NLVPMVATV", "AAAAAAAA"] * 2,
                             ["AR"] * 8, ["DC"] * 8)
    mask = numpy.array([True, False, True, False, True, False, True, False])
    model.fit(inputs, [1, 0] * 4, validation_mask=mask, seed=42,
              verbose=-1, progress_print_interval=None)
    info = model.fit_info[-1]["initialization"]
    assert info["applied"] and info["method"] == "lsuv_pre"
    assert len(info["calibration_fit_input_rows"]) == 3
    assert not mask[info["calibration_fit_input_rows"]].any()
    predictor = Class1ProcessingPredictor([model])
    predictor.save(str(tmp_path))
    loaded = Class1ProcessingPredictor.load(str(tmp_path))
    assert loaded.models[0].fit_info[-1]["initialization"] == info
    numpy.testing.assert_allclose(model.predict_encoded(inputs), loaded.models[0].predict_encoded(inputs))
