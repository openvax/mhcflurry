"""
Custom loss functions.

Supports training a regressor on data that includes inequalities (e.g. x < 100).

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

from keras import backend as K

import pandas
import numpy

LOSSES = {}


def encode_y(y, inequalities=None):
    y = numpy.array(y, dtype="float32")
    if y.isnan().any():
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
            '<': 2,
            '>': 4,
        }).values
        if offsets.isnan().any():
            raise ValueError("Invalid inequality. Must be =, <, or >")
        encoded = y + offsets
    assert not encoded.isnan().any()
    return encoded


def mse_with_ineqalities(y_true, y_pred):
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
LOSSES["mse_with_ineqalities"] = mse_with_ineqalities