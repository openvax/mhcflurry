"""
PyTorch loss functions for mhcflurry.

Supports inequality constraints where training data includes (=), (<), and (>)
relationships. For inequality constraints, penalization is applied only when
predictions violate the constraint.
"""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEWithInequalities(nn.Module):
    """
    MSE loss with inequality support.

    y_true is encoded as follows:
      - [0, 1]: equality constraint, standard MSE
      - [2, 3]: greater-than constraint (value = y_true - 2), penalize if pred < value
      - [4, 5]: less-than constraint (value = y_true - 4), penalize if pred > value
    """
    supports_inequalities = True
    supports_multiple_outputs = False

    @staticmethod
    def encode_y(y, inequalities=None):
        """
        Encode targets with inequality information.

        Parameters
        ----------
        y : array-like
            Target values in [0, 1]
        inequalities : array-like of str, optional
            One of "=", ">", "<" for each target

        Returns
        -------
        numpy.ndarray
        """
        y = numpy.array(y, dtype=numpy.float32)
        if numpy.isnan(y).any():
            raise ValueError("y contains NaN")
        if (y < 0).any() or (y > 1).any():
            raise ValueError("Targets must be in [0, 1] for MSEWithInequalities")
        if inequalities is None:
            return y
        if len(inequalities) != len(y):
            raise ValueError("inequalities must have same length as y")
        for ineq in inequalities:
            if ineq not in {"=", ">", "<"}:
                raise ValueError("Inequalities must be one of '=', '>', '<'")
        offsets = numpy.array([
            {'=': 0, '>': 2, '<': 4}[ineq] for ineq in inequalities
        ], dtype=numpy.float32)
        encoded = y + offsets
        assert not numpy.isnan(encoded).any()
        return encoded

    def forward(self, y_pred, y_true, sample_weights=None):
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions, shape (batch,) or (batch, 1)
        y_true : torch.Tensor
            Encoded targets, shape (batch,) or (batch, 1)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        # Handle (=) inequalities: 0 <= y_true <= 1
        diff1 = y_pred - y_true
        diff1 = diff1 * (y_true >= 0.0).float() * (y_true <= 1.0).float()

        # Handle (>) inequalities: 2 <= y_true <= 3
        # Penalize only if pred < threshold (diff < 0)
        diff2 = y_pred - (y_true - 2.0)
        diff2 = diff2 * (y_true >= 2.0).float() * (y_true <= 3.0).float()
        diff2 = diff2 * (diff2 < 0.0).float()

        # Handle (<) inequalities: y_true >= 4
        # Penalize only if pred > threshold (diff > 0)
        diff3 = y_pred - (y_true - 4.0)
        diff3 = diff3 * (y_true >= 4.0).float()
        diff3 = diff3 * (diff3 > 0.0).float()

        per_sample = diff1.square() + diff2.square() + diff3.square()
        if sample_weights is None:
            denominator = torch.clamp(
                (y_true != 2.0).float().sum(), min=1.0
            )
            return per_sample.sum() / denominator
        sample_weights = sample_weights.reshape(-1).to(per_sample.device)
        mask = (y_true != 2.0).float()
        denominator = torch.clamp((sample_weights * mask).sum(), min=1.0)
        return (per_sample * sample_weights).sum() / denominator


class MSEWithInequalitiesAndMultipleOutputs(nn.Module):
    """
    MSE loss with inequality and multiple output support.

    Extends MSEWithInequalities by encoding the output index into the target:
    encoded_target = inequality_encoded_value + output_index * 10
    """
    supports_inequalities = True
    supports_multiple_outputs = True

    @staticmethod
    def encode_y(y, inequalities=None, output_indices=None):
        """
        Encode targets with inequality and output index information.

        Parameters
        ----------
        y : array-like
            Target values in [0, 1]
        inequalities : array-like of str, optional
            One of "=", ">", "<" for each target
        output_indices : array-like of int, optional
            Output index for each target

        Returns
        -------
        numpy.ndarray
        """
        encoded = MSEWithInequalities.encode_y(y, inequalities)
        if output_indices is not None:
            output_indices = numpy.array(output_indices)
            if output_indices.shape != (len(encoded),):
                raise ValueError(
                    "Expected output_indices to have shape %s not %s"
                    % ((len(encoded),), output_indices.shape)
                )
            if (output_indices < 0).any():
                raise ValueError("Invalid output indices: %s" % output_indices)
            encoded = encoded + output_indices.astype(numpy.float32) * 10
        return encoded

    def forward(self, y_pred, y_true, sample_weights=None):
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions, shape (batch, num_outputs)
        y_true : torch.Tensor
            Encoded targets, shape (batch,) or (batch, 1)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        y_true = y_true.reshape(-1)
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)

        # Decode output indices
        output_indices = (y_true / 10.0).long()
        inequality_encoded = y_true - output_indices.float() * 10.0

        # Select the relevant output for each sample
        batch_indices = torch.arange(len(y_true), device=y_pred.device)
        output_indices_clamped = output_indices.clamp(0, y_pred.shape[1] - 1)
        selected_pred = y_pred[batch_indices, output_indices_clamped]

        # Apply MSEWithInequalities logic on selected predictions
        y_t = inequality_encoded
        y_p = selected_pred

        # Handle (=) inequalities
        diff1 = y_p - y_t
        diff1 = diff1 * (y_t >= 0.0).float() * (y_t <= 1.0).float()

        # Handle (>) inequalities
        diff2 = y_p - (y_t - 2.0)
        diff2 = diff2 * (y_t >= 2.0).float() * (y_t <= 3.0).float()
        diff2 = diff2 * (diff2 < 0.0).float()

        # Handle (<) inequalities
        diff3 = y_p - (y_t - 4.0)
        diff3 = diff3 * (y_t >= 4.0).float()
        diff3 = diff3 * (diff3 > 0.0).float()

        per_sample = diff1.square() + diff2.square() + diff3.square()
        if sample_weights is None:
            denominator = torch.clamp(
                (y_t != 2.0).float().sum(), min=1.0
            )
            return per_sample.sum() / denominator
        sample_weights = sample_weights.reshape(-1).to(per_sample.device)
        mask = (y_t != 2.0).float()
        denominator = torch.clamp((sample_weights * mask).sum(), min=1.0)
        return (per_sample * sample_weights).sum() / denominator


class MultiallelicMassSpecLoss(nn.Module):
    """
    Loss function for multiallelic mass spectrometry data.

    For each (hit, decoy) pair, penalizes when any decoy allele prediction
    exceeds the best hit allele prediction by more than delta.

    y_true encoding:
      - 1.0: hit (positive)
      - 0.0: decoy (negative)
      - -1.0: ignored
    """
    supports_inequalities = True
    supports_multiple_outputs = False

    def __init__(self, delta=0.2, multiplier=1.0):
        super(MultiallelicMassSpecLoss, self).__init__()
        self.delta = delta
        self.multiplier = multiplier

    @staticmethod
    def encode_y(y):
        """Encode y (no-op for this loss)."""
        y = numpy.array(y, dtype=numpy.float32)
        assert numpy.isin(y, [-1.0, 0.0, 1.0]).all()
        return y

    def forward(self, y_pred, y_true, sample_weights=None):
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions, shape (batch, num_alleles)
        y_true : torch.Tensor
            Labels, shape (batch,) or (batch, 1)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        y_true = y_true.reshape(-1)

        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)

        # Get hit and decoy masks
        hit_mask = (y_true == 1.0)
        decoy_mask = (y_true == 0.0)

        num_hits = hit_mask.sum().item()
        num_decoys = decoy_mask.sum().item()

        if num_hits == 0 or num_decoys == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        num_alleles = y_pred.shape[1]

        # Best allele prediction for each hit: (num_hits,)
        hit_preds = y_pred[hit_mask]
        hit_max = hit_preds.max(dim=1).values  # (num_hits,)

        # All decoy predictions: (num_decoys, num_alleles)
        decoy_preds = y_pred[decoy_mask]

        # Compute pairwise terms:
        # For each (decoy, allele, hit): max(0, decoy_pred - hit_max + delta)^2
        # decoy_preds: (num_decoys, num_alleles) -> (num_decoys, num_alleles, 1)
        # hit_max: (num_hits,) -> (1, 1, num_hits)
        term = decoy_preds.unsqueeze(2) - hit_max.unsqueeze(0).unsqueeze(0) + self.delta
        penalty = torch.clamp(term, min=0.0).square()

        denominator = num_hits * num_decoys * num_alleles
        result = self.multiplier * penalty.sum() / denominator

        return result


class StandardLoss(nn.Module):
    """
    Wrapper for standard PyTorch loss functions (MSE, MAE, etc).
    """
    supports_inequalities = False
    supports_multiple_outputs = False

    def __init__(self, loss_name="mse"):
        super(StandardLoss, self).__init__()
        self.loss_name = loss_name
        if loss_name == "mse":
            self._loss_fn = nn.MSELoss()
        elif loss_name == "mae":
            self._loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown standard loss: {loss_name}")

    @staticmethod
    def encode_y(y):
        """Encode y (simple cast to float32)."""
        return numpy.array(y, dtype=numpy.float32)

    def forward(self, y_pred, y_true, sample_weights=None):
        """
        Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
        y_true : torch.Tensor
        sample_weights : torch.Tensor | None
            Optional per-example weights.

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        if sample_weights is None:
            return self._loss_fn(y_pred, y_true)
        if self.loss_name == "mse":
            losses = F.mse_loss(y_pred, y_true, reduction="none")
        elif self.loss_name == "mae":
            losses = F.l1_loss(y_pred, y_true, reduction="none")
        else:
            losses = self._loss_fn(y_pred, y_true)
        if losses.dim() > 1:
            losses = losses.view(losses.shape[0], -1).mean(dim=1)
        sample_weights = sample_weights.reshape(-1).to(losses.device)
        denominator = torch.clamp(sample_weights.sum(), min=1.0)
        return (losses * sample_weights).sum() / denominator


# Registry of custom losses
_CUSTOM_LOSSES = {
    'mse_with_inequalities': MSEWithInequalities,
    'mse_with_inequalities_and_multiple_outputs': MSEWithInequalitiesAndMultipleOutputs,
    'multiallelic_mass_spec_loss': MultiallelicMassSpecLoss,
}


def get_pytorch_loss(name):
    """
    Get a PyTorch loss object by name.

    Parameters
    ----------
    name : str
        Loss name. Prefix with "custom:" for custom losses,
        otherwise a standard loss name like "mse".

    Returns
    -------
    nn.Module
        Loss module with encode_y, supports_inequalities,
        and supports_multiple_outputs attributes.
    """
    if name.startswith("custom:"):
        custom_name = name.replace("custom:", "")
        if custom_name not in _CUSTOM_LOSSES:
            raise ValueError(
                f"No such custom loss: {name}. "
                f"Supported: {', '.join('custom:' + k for k in _CUSTOM_LOSSES)}"
            )
        return _CUSTOM_LOSSES[custom_name]()
    return StandardLoss(name)
