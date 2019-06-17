from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal

import numpy

numpy.random.seed(0)

import logging
logging.getLogger('tensorflow').disabled = True

import keras.backend as K

from mhcflurry.custom_loss import CUSTOM_LOSSES


def evaluate_loss(loss, y_true, y_pred):
    y_true = numpy.array(y_true)
    y_pred = numpy.array(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((len(y_pred), 1))
    if y_true.ndim == 1:
        y_true = y_true.reshape((len(y_true), 1))

    assert y_true.ndim == 2
    assert y_pred.ndim == 2

    if K.backend() == "tensorflow":
        session = K.get_session()
        y_true_var = K.constant(y_true, name="y_true")
        y_pred_var = K.constant(y_pred, name="y_pred")
        result = loss(y_true_var, y_pred_var)
        return result.eval(session=session)
    elif K.backend() == "theano":
        y_true_var = K.constant(y_true, name="y_true")
        y_pred_var = K.constant(y_pred, name="y_pred")
        result = loss(y_true_var, y_pred_var)
        return result.eval()
    else:
        raise ValueError("Unsupported backend: %s" % K.backend())


def test_mse_with_inequalities(loss_obj=CUSTOM_LOSSES['mse_with_inequalities']):
    y_values = [0.0, 0.5, 0.8, 1.0]

    adjusted_y = loss_obj.encode_y(y_values)
    print(adjusted_y)
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
    print(loss0)
    eq_(loss0, 0.0)

    adjusted_y = loss_obj.encode_y(y_values, [">", ">", ">", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
    eq_(loss0, 0.0)

    adjusted_y = loss_obj.encode_y(y_values, ["<", "<", "<", "<"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
    eq_(loss0, 0.0)

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
    eq_(loss0, 0.0)

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.0, 0.4, 0.8, 1.0])
    eq_(loss0, 0.0)

    adjusted_y = loss_obj.encode_y(y_values, [">", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.4, 0.9, 1.0])
    eq_(loss0, 0.0)

    adjusted_y = loss_obj.encode_y(y_values, [">", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert_greater(loss0, 0.0)

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", ">", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert_almost_equal(loss0, 0.02 / 4)

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert_almost_equal(loss0, 0.03 / 4)


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
    assert_almost_equal(loss0, 0.0)

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
    assert_almost_equal(loss0, 0.02 / 4)

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
    assert_almost_equal(loss0, 0.01 / 4)

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
    assert_almost_equal(loss0, 0.02 / 4)

