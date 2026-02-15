"""
Custom loss functions.

For losses supporting inequalities, each training data point is associated with
one of (=), (<), or (>). For e.g. (>) inequalities, penalization is applied only
if the prediction is less than the given value.

This module now delegates to pytorch_losses.py for the actual loss implementations.
"""
from __future__ import division
import numpy

# Import PyTorch implementations
from .pytorch_losses import (
    MSEWithInequalities as PyTorchMSEWithInequalities,
    MSEWithInequalitiesAndMultipleOutputs as PyTorchMSEWithInequalitiesAndMultipleOutputs,
    MultiallelicMassSpecLoss as PyTorchMultiallelicMassSpecLoss,
    StandardLoss as PyTorchStandardLoss,
)

CUSTOM_LOSSES = {}


def get_loss(name):
    """
    Get a custom_loss.Loss instance by name.

    Parameters
    ----------
    name : string

    Returns
    -------
    custom_loss.Loss
    """
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
    """
    Thin wrapper to keep track of neural network loss functions, which could
    be custom or baked into PyTorch.

    Each subclass or instance should define these properties/methods:
    - name : string
    - loss : callable
        This is the PyTorch loss function
    - encode_y : numpy.ndarray -> numpy.ndarray
        Transformation to apply to regression target before fitting
    """
    def __init__(self, name=None):
        self.name = name if name else self.name  # use name from class instance

    def __str__(self):
        return "<Loss: %s>" % self.name

    def loss(self, y_true, y_pred):
        raise NotImplementedError()

    def get_keras_loss(self, reduction="sum_over_batch_size"):
        """
        Backward-compatible accessor from the TF/Keras backend era.

        Parameters
        ----------
        reduction : string
            Ignored. Kept for API compatibility.
        """
        del reduction  # unused legacy argument
        return self.loss


class StandardKerasLoss(Loss):
    """
    A standard loss function such as MSE.
    """
    supports_inequalities = False
    supports_multiple_outputs = False

    def __init__(self, loss_name="mse"):
        self._pytorch_loss = PyTorchStandardLoss(loss_name)
        self.loss = loss_name
        Loss.__init__(self, loss_name)

    @staticmethod
    def encode_y(y):
        return numpy.array(y, dtype=numpy.float32)


class TransformPredictionsLossWrapper(Loss):
    """
    Wrapper that applies an arbitrary transform to y_pred before calling an
    underlying loss function.

    The y_pred_transform function should be a tensor -> tensor function.
    """
    def __init__(
            self,
            loss,
            y_pred_transform=None):
        self.wrapped_loss = loss
        self.name = "transformed_%s" % loss.name
        self.y_pred_transform = y_pred_transform
        self.supports_inequalities = loss.supports_inequalities
        self.supports_multiple_outputs = loss.supports_multiple_outputs

    def encode_y(self, *args, **kwargs):
        return self.wrapped_loss.encode_y(*args, **kwargs)

    def loss(self, y_true, y_pred):
        y_pred_transformed = self.y_pred_transform(y_pred)
        return self.wrapped_loss.loss(y_true, y_pred_transformed)


class MSEWithInequalities(Loss):
    """
    Supports training a regression model on data that includes inequalities
    (e.g. x < 100). Mean square error is used as the loss for elements with
    an (=) inequality. For elements with e.g. a (> 0.5) inequality, then the loss
    for that element is (y - 0.5)^2 (standard MSE) if y < 500 and 0 otherwise.

    This loss assumes that the normal range for y_true and y_pred is 0 - 1. As a
    hack, the implementation uses other intervals for y_pred to encode the
    inequality information.

    y_true is interpreted as follows:

    between 0 - 1
       Regular MSE loss is used. Penalty (y_pred - y_true)**2 is applied if
       y_pred is greater or less than y_true.

    between 2 - 3:
       Treated as a ">" inequality. Penalty (y_pred - (y_true - 2))**2 is
       applied only if y_pred is less than y_true - 2.

    between 4 - 5:
       Treated as a "<" inequality. Penalty (y_pred - (y_true - 4))**2 is
       applied only if y_pred is greater than y_true - 4.
    """
    name = "mse_with_inequalities"
    supports_inequalities = True
    supports_multiple_outputs = False

    def __init__(self):
        self._pytorch_loss = PyTorchMSEWithInequalities()

    @staticmethod
    def encode_y(y, inequalities=None):
        return PyTorchMSEWithInequalities.encode_y(y, inequalities)

    @staticmethod
    def _max_value(values):
        if hasattr(values, "detach"):
            return float(values.detach().max().item())
        return float(numpy.asarray(values).max())

    def loss(self, y_true, y_pred):
        # Support both historical Keras-style (y_true, y_pred) and current
        # PyTorch-style (y_pred, y_true) calling conventions.
        if self._max_value(y_true) <= 1.5 and self._max_value(y_pred) > 1.5:
            y_true, y_pred = y_pred, y_true
        return self._pytorch_loss(y_pred, y_true)


class MSEWithInequalitiesAndMultipleOutputs(Loss):
    """
    Loss supporting inequalities and multiple outputs.

    This loss assumes that the normal range for y_true and y_pred is 0 - 1. As a
    hack, the implementation uses other intervals for y_pred to encode the
    inequality and output-index information.

    Inequalities are encoded into the regression target as in
    the MSEWithInequalities loss.

    Multiple outputs are encoded by mapping each regression target x (after
    transforming for inequalities) using the rule x -> x + i * 10 where i is
    the output index.

    The reason for explicitly encoding multiple outputs this way (rather than
    just making the regression target a matrix instead of a vector) is that
    in our use cases we frequently have missing data in the regression target.
    This encoding gives a simple way to penalize only on (data point, output
    index) pairs that have labels.
    """
    name = "mse_with_inequalities_and_multiple_outputs"
    supports_inequalities = True
    supports_multiple_outputs = True

    def __init__(self):
        self._pytorch_loss = PyTorchMSEWithInequalitiesAndMultipleOutputs()

    @staticmethod
    def encode_y(y, inequalities=None, output_indices=None):
        return PyTorchMSEWithInequalitiesAndMultipleOutputs.encode_y(
            y, inequalities, output_indices
        )

    def loss(self, y_true, y_pred):
        # Support both historical Keras-style (y_true, y_pred) and current
        # PyTorch-style (y_pred, y_true) calling conventions.
        if (
                getattr(y_true, "ndim", None) == 2 and
                getattr(y_pred, "ndim", None) == 2 and
                y_true.shape[1] > 1 and
                y_pred.shape[1] == 1):
            y_true, y_pred = y_pred, y_true
        else:
            max_true = MSEWithInequalities._max_value(y_true)
            max_pred = MSEWithInequalities._max_value(y_pred)
            if max_true <= 1.5 and max_pred > 1.5:
                y_true, y_pred = y_pred, y_true
        return self._pytorch_loss(y_pred, y_true)


class MultiallelicMassSpecLoss(Loss):
    """
    Multiallelic mass spec loss function.
    """
    name = "multiallelic_mass_spec_loss"
    supports_inequalities = True
    supports_multiple_outputs = False

    def __init__(self, delta=0.2, multiplier=1.0):
        self.delta = delta
        self.multiplier = multiplier
        self._pytorch_loss = PyTorchMultiallelicMassSpecLoss(delta, multiplier)

    @staticmethod
    def encode_y(y):
        return PyTorchMultiallelicMassSpecLoss.encode_y(y)

    def loss(self, y_true, y_pred):
        # Support both historical Keras-style (y_true, y_pred) and current
        # PyTorch-style (y_pred, y_true) calling conventions.
        if getattr(y_true, "ndim", None) == 2 and y_true.shape[1] > 1:
            y_true, y_pred = y_pred, y_true
        return self._pytorch_loss(y_pred, y_true)


def check_shape(name, arr, expected_shape):
    """
    Raise ValueError if arr.shape != expected_shape.

    Parameters
    ----------
    name : string
        Included in error message to aid debugging
    arr : numpy.ndarray
    expected_shape : tuple of int
    """
    if arr.shape != expected_shape:
        raise ValueError("Expected %s to have shape %s not %s" % (
            name, str(expected_shape), str(arr.shape)))


# Register custom losses.
for cls in [
        MSEWithInequalities,
        MSEWithInequalitiesAndMultipleOutputs,
        MultiallelicMassSpecLoss]:
    CUSTOM_LOSSES[cls.name] = cls()
