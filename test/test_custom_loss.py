"""
Tests for custom loss functions.
"""
import pytest
from . import initialize
initialize()

from .pytest_helpers import eq_, assert_less, assert_greater, assert_almost_equal

import numpy
import torch
from mhcflurry.custom_loss import CUSTOM_LOSSES, MultiallelicMassSpecLoss

from mhcflurry.testing_utils import cleanup, startup


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    startup()
    yield
    cleanup()


def evaluate_loss(loss_obj, y_true, y_pred):
    """Evaluate a loss function with PyTorch tensors."""
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(len(y_pred), 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(len(y_true), 1)

    print("y_pred, y_true:", y_pred, y_true)

    assert y_true.ndim == 2
    assert y_pred.ndim == 2

    result = loss_obj.loss(y_pred, y_true)
    return result.item()


def test_mse_with_inequalities(loss_obj=None):
    """Test MSE with inequalities loss function."""
    if loss_obj is None:
        loss_obj = CUSTOM_LOSSES['mse_with_inequalities']

    y_values = [0.0, 0.5, 0.8, 1.0]

    adjusted_y = loss_obj.encode_y(y_values)
    print(adjusted_y)
    loss0 = evaluate_loss(loss_obj, adjusted_y, y_values)
    print(loss0)
    assert abs(loss0) < 1e-6, f"Expected 0, got {loss0}"

    adjusted_y = loss_obj.encode_y(y_values, [">", ">", ">", ">"])
    loss0 = evaluate_loss(loss_obj, adjusted_y, y_values)
    assert abs(loss0) < 1e-6, f"Expected 0, got {loss0}"

    adjusted_y = loss_obj.encode_y(y_values, ["<", "<", "<", "<"])
    loss0 = evaluate_loss(loss_obj, adjusted_y, y_values)
    assert abs(loss0) < 1e-6, f"Expected 0, got {loss0}"

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj, adjusted_y, y_values)
    assert abs(loss0) < 1e-6, f"Expected 0, got {loss0}"

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj, adjusted_y, [0.0, 0.4, 0.8, 1.0])
    assert abs(loss0) < 1e-6, f"Expected 0, got {loss0}"

    adjusted_y = loss_obj.encode_y(y_values, [">", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj, adjusted_y, [0.1, 0.4, 0.9, 1.0])
    assert abs(loss0) < 1e-6, f"Expected 0, got {loss0}"

    adjusted_y = loss_obj.encode_y(y_values, [">", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert_greater(loss0, 0.0)

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert_almost_equal(loss0, 0.02 / 4, places=5)

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert_almost_equal(loss0, 0.03 / 4, places=5)


def test_mse_with_inequalities_and_multiple_outputs():
    """Test MSE with inequalities and multiple outputs loss function."""
    loss_obj = CUSTOM_LOSSES['mse_with_inequalities_and_multiple_outputs']
    test_mse_with_inequalities(loss_obj)

    y_values = [0.0, 0.5, 0.8, 1.0]
    adjusted_y = loss_obj.encode_y(
        y_values, output_indices=[0, 1, 1, 1])
    loss0 = evaluate_loss(
        loss_obj,
        adjusted_y,
        [
            [0.0, 1000],
            [2000, 0.5],
            [3000, 0.8],
            [4000, 1.0],
        ])
    assert_almost_equal(loss0, 0.0, places=5)

    y_values = [0.0, 0.5, 0.8, 1.0]
    adjusted_y = loss_obj.encode_y(
        y_values, output_indices=[0, 1, 1, 0])
    loss0 = evaluate_loss(
        loss_obj,
        adjusted_y,
        [
            [0.1, 1000],
            [2000, 0.6],
            [3000, 0.8],
            [1.0, 4000],
        ])
    assert_almost_equal(loss0, 0.02 / 4, places=5)

    y_values = [0.0, 0.5, 0.8, 1.0]
    adjusted_y = loss_obj.encode_y(
        y_values, output_indices=[0, 1, 1, 0], inequalities=["=", ">", "<", "<"])
    loss0 = evaluate_loss(
        loss_obj,
        adjusted_y,
        [
            [0.1, 1000],
            [2000, 0.6],
            [3000, 0.8],
            [1.0, 4000],
        ])
    assert_almost_equal(loss0, 0.01 / 4, places=5)

    y_values = [0.0, 0.5, 0.8, 1.0]
    adjusted_y = loss_obj.encode_y(
        y_values, output_indices=[0, 1, 1, 0], inequalities=["=", "<", "<", "<"])
    loss0 = evaluate_loss(
        loss_obj,
        adjusted_y,
        [
            [0.1, 1000],
            [2000, 0.6],
            [3000, 0.8],
            [1.0, 4000],
        ])
    assert_almost_equal(loss0, 0.02 / 4, places=5)


def test_multiallelic_mass_spec_loss():
    """Test multiallelic mass spec loss function."""
    for delta in [0.0, 0.3]:
        print("delta", delta)
        # Hit labels
        y_true = [
            1.0,
            0.0,
            1.0,
            -1.0,  # ignored
            1.0,
            0.0,
            1.0,
        ]
        y_true = numpy.array(y_true)
        y_pred = [
            [0.3, 0.7, 0.5],
            [0.2, 0.4, 0.6],
            [0.1, 0.5, 0.3],
            [0.9, 0.1, 0.2],
            [0.1, 0.7, 0.1],
            [0.8, 0.2, 0.4],
            [0.1, 0.2, 0.4],
        ]
        y_pred = numpy.array(y_pred)

        # reference implementation 1

        def smooth_max(x, alpha):
            x = numpy.array(x)
            alpha = numpy.array([alpha])
            return (x * numpy.exp(x * alpha)).sum() / (
                numpy.exp(x * alpha)).sum()

        contributions = []
        for i in range(len(y_true)):
            if y_true[i] == 1.0:
                for j in range(len(y_true)):
                    if y_true[j] == 0.0:
                        tightest_i = max(y_pred[i])
                        for k in range(y_pred.shape[1]):
                            contribution = max(
                                0, y_pred[j, k] - tightest_i + delta)**2
                            contributions.append(contribution)
        contributions = numpy.array(contributions)
        expected1 = contributions.sum() / len(contributions)

        # reference implementation 2: numpy
        pos = numpy.array([
            max(y_pred[i])
            for i in range(len(y_pred))
            if y_true[i] == 1.0
        ])

        neg = y_pred[(y_true == 0.0).astype(bool)]
        term = neg.reshape((-1, 1)) - pos + delta
        expected2 = (
                numpy.maximum(0, term)**2).sum() / (
            len(pos) * neg.shape[0] * neg.shape[1])

        numpy.testing.assert_almost_equal(expected1, expected2)

        loss_obj = MultiallelicMassSpecLoss(delta=delta)
        computed = evaluate_loss(
            loss_obj,
            y_true,
            y_pred.reshape(y_pred.shape))

        numpy.testing.assert_almost_equal(computed, expected1, 4)


def test_encode_y_basic():
    """Test basic y encoding functionality."""
    from mhcflurry.pytorch_losses import MSEWithInequalities

    # Test equality encoding
    y = [0.0, 0.5, 1.0]
    encoded = MSEWithInequalities.encode_y(y)
    numpy.testing.assert_array_equal(encoded, y)

    # Test greater than encoding (should add 2)
    encoded_gt = MSEWithInequalities.encode_y(y, [">", ">", ">"])
    numpy.testing.assert_array_equal(encoded_gt, [2.0, 2.5, 3.0])

    # Test less than encoding (should add 4)
    encoded_lt = MSEWithInequalities.encode_y(y, ["<", "<", "<"])
    numpy.testing.assert_array_equal(encoded_lt, [4.0, 4.5, 5.0])


def test_loss_gradient_flow():
    """Test that gradients flow correctly through the loss."""
    from mhcflurry.pytorch_losses import MSEWithInequalities

    loss_fn = MSEWithInequalities()

    # Create predictions that require gradients
    y_pred = torch.tensor([[0.5]], requires_grad=True)
    y_true = torch.tensor([[0.3]])  # equality

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    # Gradient should exist and be non-zero
    assert y_pred.grad is not None
    assert y_pred.grad.abs().item() > 0


def test_inequality_gradient_respects_constraint():
    """Test that gradients respect inequality constraints."""
    from mhcflurry.pytorch_losses import MSEWithInequalities

    loss_fn = MSEWithInequalities()

    # Test greater-than constraint (y_true encoded as 2 + value)
    # When pred > threshold, gradient should be 0
    y_pred = torch.tensor([[0.7]], requires_grad=True)
    y_true = torch.tensor([[2.5]])  # > 0.5

    loss = loss_fn(y_pred, y_true)
    loss.backward()

    # Gradient should be 0 since pred (0.7) > threshold (0.5)
    assert abs(y_pred.grad.item()) < 1e-6

    # When pred < threshold, gradient should be non-zero
    y_pred2 = torch.tensor([[0.3]], requires_grad=True)
    y_true2 = torch.tensor([[2.5]])  # > 0.5

    loss2 = loss_fn(y_pred2, y_true2)
    loss2.backward()

    # Gradient should be non-zero since pred (0.3) < threshold (0.5)
    assert abs(y_pred2.grad.item()) > 0
