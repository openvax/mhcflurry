from nose.tools import eq_, assert_less, assert_greater, assert_almost_equal

import numpy

numpy.random.seed(0)

import logging
logging.getLogger('tensorflow').disabled = True

import keras.backend as K

from mhcflurry.custom_loss import CUSTOM_LOSSES


def evaluate_loss(loss, y_true, y_pred):
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


def test_mse_with_inequalities():

    loss_obj = CUSTOM_LOSSES['mse_with_inequalities']

    y_values = [0.0, 0.5, 0.8, 1.0]

    adjusted_y = loss_obj.encode_y(y_values)
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, y_values)
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
    assert_almost_equal(loss0, 0.02)

    adjusted_y = loss_obj.encode_y(y_values, ["=", "<", "=", ">"])
    loss0 = evaluate_loss(loss_obj.loss, adjusted_y, [0.1, 0.6, 0.9, 1.0])
    assert_almost_equal(loss0, 0.03)

