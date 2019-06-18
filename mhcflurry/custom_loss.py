"""
Custom loss functions.

For losses supporting inequalities, each training data point is associated with
one of (=), (<), or (>). For e.g. (>) inequalities, penalization is applied only
if the prediction is less than the given value.
"""
from __future__ import division
import pandas
import numpy
from numpy import isnan, array

CUSTOM_LOSSES = {}


def get_loss(name):
    if name.startswith("custom:"):
        try:
            custom_loss = CUSTOM_LOSSES[name.replace("custom:", "")]
        except KeyError:
            raise ValueError(
                "No such custom loss: %s. Supported losses are: %s" % (
                    name,
                    ", ".join([
                        "custom:" + loss_name for loss_name in CUSTOM_LOSSES
                    ])))
        return custom_loss
    return StandardKerasLoss(name)


class Loss(object):
    def __init__(self, name=None):
        self.name = name if name else self.name  # use name from class instance

    def __str__(self):
        return "<Loss: %s>" % self.name


class StandardKerasLoss(Loss):
    supports_inequalities = False
    supports_multiple_outputs = False

    def __init__(self, loss_name="mse"):
        self.loss = loss_name
        Loss.__init__(self, loss_name)

    @staticmethod
    def encode_y(y):
        return y


class MSEWithInequalities(Loss):
    """
    Supports training a regressor on data that includes inequalities
    (e.g. x < 100). Mean square error is used as the loss for elements with
    an (=) inequality. For elements with e.g. a (> 0.5) inequality, then the loss
    for that element is (y - 0.5)^2 (standard MSE) if y < 500 and 0 otherwise.

    This loss assumes that the normal range for y_true and y_pred is 0 - 1. As a
    hack, the implementation uses other intervals for y_pred to encode the
    inequality information.

    y_true is interpreted as follows:

    between 0 - 1
       Regular MSE loss is used. Penality (y_pred - y_true)**2 is applied if
       y_pred is greater or less than y_true.

    between 2 - 3:
       Treated as a "<" inequality. Penality (y_pred - (y_true - 2))**2 is
       applied only if y_pred is greater than y_true - 2.

    between 4 - 5:
       Treated as a ">" inequality. Penality (y_pred - (y_true - 4))**2 is
       applied only if y_pred is less than y_true - 4.
    """
    name = "mse_with_inequalities"
    supports_inequalities = True
    supports_multiple_outputs = False

    @staticmethod
    def encode_y(y, inequalities=None):
        y = array(y, dtype="float32")
        if isnan(y).any():
            raise ValueError("y contains NaN")
        if (y > 1.0).any():
            raise ValueError("y contains values > 1.0")
        if (y < 0.0).any():
            raise ValueError("y contains values < 0.0")

        if inequalities is None:
            encoded = y
        else:
            offsets = pandas.Series(inequalities).map({
                '=': 0,
                '>': 2,
                '<': 4,
            }).values
            if isnan(offsets).any():
                raise ValueError("Invalid inequality. Must be =, <, or >")
            encoded = y + offsets
        assert not isnan(encoded).any()
        return encoded

    @staticmethod
    def loss(y_true, y_pred):
        # We always delay import of Keras so that mhcflurry can be imported initially
        # without tensorflow debug output, etc.
        from keras import backend as K

        # Handle (=) inequalities
        diff1 = y_pred - y_true
        diff1 *= K.cast(y_true >= 0.0, "float32")
        diff1 *= K.cast(y_true <= 1.0, "float32")

        # Handle (>) inequalities
        diff2 = y_pred - (y_true - 2.0)
        diff2 *= K.cast(y_true >= 2.0, "float32")
        diff2 *= K.cast(y_true <= 3.0, "float32")
        diff2 *= K.cast(diff2 < 0.0, "float32")

        # Handle (<) inequalities
        diff3 = y_pred - (y_true - 4.0)
        diff3 *= K.cast(y_true >= 4.0, "float32")
        diff3 *= K.cast(diff3 > 0.0, "float32")

        result = (
            K.sum(K.square(diff1)) +
            K.sum(K.square(diff2)) +
            K.sum(K.square(diff3))) / K.cast(K.shape(y_pred)[0], "float32")

        return result


class MSEWithInequalitiesAndMultipleOutputs(Loss):
    name = "mse_with_inequalities_and_multiple_outputs"
    supports_inequalities = True
    supports_multiple_outputs = True

    @staticmethod
    def encode_y(y, inequalities=None, output_indices=None):
        y = array(y, dtype="float32")
        if isnan(y).any():
            raise ValueError("y contains NaN")
        if (y > 1.0).any():
            raise ValueError("y contains values > 1.0")
        if (y < 0.0).any():
            raise ValueError("y contains values < 0.0")

        encoded = MSEWithInequalities.encode_y(
            y, inequalities=inequalities)

        if output_indices is not None:
            output_indices = numpy.array(output_indices)
            check_shape("output_indices", output_indices, (len(encoded),))
            if (output_indices < 0).any():
                raise ValueError("Invalid output indices: ", output_indices)

            encoded += output_indices * 10

        return encoded

    @staticmethod
    def loss(y_true, y_pred):
        from keras import backend as K

        y_true = K.flatten(y_true)

        output_indices = y_true // 10
        updated_y_true = y_true - (10 * output_indices)

        # We index into y_pred using flattened indices since Keras backend
        # supports gather but has no equivalent of tf.gather_nd:
        ordinals = K.arange(K.shape(y_true)[0])
        flattened_indices = (
            ordinals * y_pred.shape[1] + K.cast(output_indices, "int32"))
        updated_y_pred = K.gather(K.flatten(y_pred), flattened_indices)

        # Alternative implementation using tensorflow, which could be used if
        # we drop support for other backends:
        # import tensorflow as tf
        # indexer = K.stack([
        #     ordinals,
        #     K.cast(output_indices, "int32")
        # ], axis=-1)
        #updated_y_pred = tf.gather_nd(y_pred, indexer)

        return MSEWithInequalities.loss(updated_y_true, updated_y_pred)


def check_shape(name, arr, expected_shape):
    if arr.shape != expected_shape:
        raise ValueError("Expected %s to have shape %s not %s" % (
            name, str(expected_shape), str(arr.shape)))


# Register custom losses.
for cls in [MSEWithInequalities, MSEWithInequalitiesAndMultipleOutputs]:
    CUSTOM_LOSSES[cls.name] = cls()