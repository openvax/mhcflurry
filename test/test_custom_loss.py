from . import initialize
initialize()

import numpy
import tensorflow as tf
import pytest
from mhcflurry.custom_loss import CUSTOM_LOSSES, MultiallelicMassSpecLoss

from mhcflurry.testing_utils import cleanup, startup

pytest.fixture(autouse=True, scope="module")
def setup_module():
    startup()
    yield
    cleanup()

def evaluate_loss(loss, y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype='float32', name='y_true')
    y_pred = tf.convert_to_tensor(y_pred, dtype='float32', name='y_pred')
    if y_pred.ndim == 1:
        y_pred = tf.reshape(y_pred, (len(y_pred), 1))
    if y_true.ndim == 1:
        y_true = tf.reshape(y_true, (len(y_true), 1))

    print("y_pred, y_true:", y_pred, y_true)

    assert y_true.ndim == 2
    assert y_pred.ndim == 2

    result = loss(y_true, y_pred)
    return result.numpy()


def test_mse_with_inequalities(loss_obj=CUSTOM_LOSSES['mse_with_inequalities']):
    y_values = [0.0, 0.5, 0.8, 1.0]

    adjusted_y = loss_obj.encode_y(y_values)
    print(adjusted_y)
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
    print(loss0)
    assert loss0 == 0.0

    adjusted_y = loss_obj.encode_y(y_values, [">", ">", ">", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
    assert loss0 == 0.0

    adjusted_y = loss_obj.encode_y(y_values, ["<", "<", "<", "<"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
    assert loss0 == 0.0

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
    assert loss0 == 0.0

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.0, 0.4, 0.8, 1.0])
    assert loss0 == 0.0

    adjusted_y = loss_obj.encode_y(y_values, [">", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.4, 0.9, 1.0])
    assert loss0 == 0.0

    adjusted_y = loss_obj.encode_y(y_values, [">", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert loss0 > 0.0

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert loss0 == pytest.approx(0.02 / 4, abs=1e-7)

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert loss0 == pytest.approx(0.03 / 4, abs=1e-7)


def test_mse_with_inequalities_and_multiple_outputs():
    loss_obj = CUSTOM_LOSSES['mse_with_inequalities_and_multiple_outputs']
    test_mse_with_inequalities(loss_obj)

    y_values = [0.0, 0.5, 0.8, 1.0]
    adjusted_y = loss_obj.encode_y(
        y_values, output_indices=[0, 1, 1, 1])
    loss0 = evaluate_loss(
        loss_obj.loss,
        adjusted_y,
        [
            [0.0, 1000],
            [2000, 0.5],
            [3000, 0.8],
            [4000, 1.0],
        ])
    assert loss0 == pytest.approx(0.0, abs=1e-7)

    y_values = [0.0, 0.5, 0.8, 1.0]
    adjusted_y = loss_obj.encode_y(
        y_values, output_indices=[0, 1, 1, 0])
    loss0 = evaluate_loss(
        loss_obj.loss,
        adjusted_y,
        [
            [0.1, 1000],
            [2000, 0.6],
            [3000, 0.8],
            [1.0, 4000],
        ])
    assert loss0 == pytest.approx(0.02 / 4, abs=1e-7)

    y_values = [0.0, 0.5, 0.8, 1.0]
    adjusted_y = loss_obj.encode_y(
        y_values, output_indices=[0, 1, 1, 0], inequalities=["=", ">", "<", "<"])
    loss0 = evaluate_loss(
        loss_obj.loss,
        adjusted_y,
        [
            [0.1, 1000],
            [2000, 0.6],
            [3000, 0.8],
            [1.0, 4000],
        ])
    assert loss0 == pytest.approx(0.01 / 4, abs=1e-7)

    y_values = [0.0, 0.5, 0.8, 1.0]
    adjusted_y = loss_obj.encode_y(
        y_values, output_indices=[0, 1, 1, 0], inequalities=["=", "<", "<", "<"])
    loss0 = evaluate_loss(
        loss_obj.loss,
        adjusted_y,
        [
            [0.1, 1000],
            [2000, 0.6],
            [3000, 0.8],
            [1.0, 4000],
        ])
    assert loss0 == pytest.approx(0.02 / 4, abs=1e-7)


def test_multiallelic_mass_spec_loss():
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

        computed = evaluate_loss(
            MultiallelicMassSpecLoss(delta=delta).loss,
            y_true,
            y_pred.reshape(y_pred.shape))

        numpy.testing.assert_almost_equal(computed, expected1, 4)
