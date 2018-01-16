"""
Custom loss functions.

For losses supporting inequalities, each training data point is associated with
one of (=), (<), or (>). For e.g. (>) inequalities, penalization is applied only
if the prediction is less than the given value.
"""

import pandas
from numpy import isnan, array

CUSTOM_LOSSES = {}


class MSEWithInequalities(object):
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

        return (
            K.sum(K.square(diff1), axis=-1) +
            K.sum(K.square(diff2), axis=-1) +
            K.sum(K.square(diff3), axis=-1))

# Register custom losses.
for cls in [MSEWithInequalities]:
    CUSTOM_LOSSES[cls.name] = cls()