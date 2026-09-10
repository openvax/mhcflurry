"""Explicit hidden-layer initialization for processing recipe experiments.

LSUV follows orthogonal initialization with sequential variance adjustment
(Mishkin and Matas, https://arxiv.org/abs/1511.06422). Scalar processing heads
and the final aggregation/gating weights are deliberately not reinitialized.
"""

import math

import torch

from .data_dependent_weights_initialization import svd_orthonormal


METHODS = ("none", "orthogonal", "lsuv_pre", "lsuv_post")


def processing_hidden_layers(model):
    """Return eligible named affine layers in the actual forward order."""
    result = []
    if model.cleavage_boundary_enabled:
        result += [("n_boundary_hidden", model.n_boundary_hidden),
                   ("c_boundary_hidden", model.c_boundary_hidden)]
    result.append(("conv1", model.conv1))
    for prefix in ("n_flank_post_convs", "c_flank_post_convs"):
        result += [("%s.%d" % (prefix, index), layer)
                   for index, layer in enumerate(getattr(model, prefix)[:-1])]
    return result


def processing_activation_variance(model, layer, inputs, post_activation=False):
    """Measure a hidden output, excluding positions beyond each input sequence.

    Post-activation means the configured activation immediately after the
    affine layer, before any normalization/dropout. Missing-context X tokens
    inside configured windows are included; maximum-length tail padding is not.
    The caller owns eval mode. Hooks are removed even if forward fails.
    """
    captured = []

    def capture(module, args, output):
        value = model.conv_activation(output) if post_activation else output
        if value.ndim == 3:
            lengths = inputs["peptide_length"].reshape(-1)
            if not model.cleavage_boundary_enabled:
                lengths = lengths + model.n_flank_length + model.c_flank_length
            positions = torch.arange(value.shape[2], device=value.device)
            mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
            value = value.transpose(1, 2)[mask]
        captured.append(value.var(unbiased=False).item())

    handle = layer.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()
    if len(captured) != 1 or not math.isfinite(captured[0]):
        raise ValueError("Initialization requires exactly one finite hidden-layer output")
    return captured[0]


def initialize_processing_network(model, inputs, method="none", margin=0.05, max_iter=32):
    """Initialize eligible hidden layers, with auditable pre/post-LSUV results.

    Parameters
    ----------
    model : Class1ProcessingModel
        Fresh eager network. Output/scalar-head parameters remain untouched.
    inputs : dict
        Fixed calibration batch containing training rows only, on model device.
    method : {"none", "orthogonal", "lsuv_pre", "lsuv_post"}
        Explicit initialization policy. None is an exact no-op, including RNG.
    margin : float
        Maximum absolute deviation of the measured variance from one.
    max_iter : int
        Maximum scaling iterations per layer; failure is explicit.

    Returns
    -------
    dict
        Effective policy, variance mask, and per-layer initialization diagnostics.
    """
    if method not in METHODS:
        raise ValueError("Unknown processing initialization method: %s" % method)
    if not 0 < margin < 1 or max_iter < 1:
        raise ValueError("Invalid processing LSUV margin or iteration limit")
    report = {"method": method, "layers": [], "variance_mask": "configured_context_without_tail_padding"}
    if method == "none":
        return report
    if not len(inputs["peptide_length"]):
        raise ValueError("Initialization requires a nonempty training calibration batch")
    training_states = [(module, module.training) for module in model.modules()]
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.eval()
    try:
        for name, layer in processing_hidden_layers(model):
            with torch.no_grad():
                weight = torch.as_tensor(svd_orthonormal(tuple(layer.weight.shape)),
                                         device=layer.weight.device, dtype=layer.weight.dtype)
                layer.weight.copy_(weight)
            record = {"layer": name, "iterations": 0}
            if method != "orthogonal":
                variance = processing_activation_variance(model, layer, inputs, method == "lsuv_post")
                record["initial_variance"] = variance
                while abs(variance - 1) > margin and record["iterations"] < max_iter:
                    if variance < 1e-14:
                        raise ValueError("Degenerate processing LSUV output: " + name)
                    with torch.no_grad():
                        layer.weight.mul_(1 / math.sqrt(variance))
                    variance = processing_activation_variance(model, layer, inputs, method == "lsuv_post")
                    record["iterations"] += 1
                record["final_variance"] = variance
                if abs(variance - 1) > margin:
                    raise ValueError("Processing LSUV did not converge: %s variance=%g" % (name, variance))
            report["layers"].append(record)
    except Exception:
        # A rejected initialization must not leave a partially transformed model.
        model.load_state_dict(original)
        raise
    finally:
        for module, training in training_states:
            module.training = training
    return report
