"""Device-resident row space for class-I affinity model training.

This module is intentionally affinity-specific. It knows about the layout
created by :meth:`Class1NeuralNetwork.fit`: random-negative examples, when
enabled, are prepended to the real affinity measurements and refreshed once
per epoch. Other model families should use their own training-data container
instead of inheriting these random-negative semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy
import torch


@dataclass
class AffinityDeviceTrainingData:
    """Device tensors for one affinity-model ``fit()`` call.

    The affinity fit loop indexes one logical row space:

    ``[random negative rows | real training rows]``.

    This container owns the torch tensors for that row space. When random
    negatives are present, ``combined_peptide`` and ``combined_allele`` hold
    the full row space; ``random_negative_x_*`` and ``x_*`` are views into
    those combined buffers. Refilling random-negative peptide rows updates the
    next epoch's batches without reallocating or recopying the real-data block.

    Peptide tensors may be either:

    * ``(N, L)`` integer amino-acid indices, consumed by the network's torch
      embedding path; or
    * ``(N, L, V)`` fixed vector encodings, widened to float32 on device when
      loaded from compact integer caches.
    """

    x_peptide: torch.Tensor
    x_allele: Optional[torch.Tensor] = None
    y_encoded: Optional[torch.Tensor] = None
    sample_weights: Optional[torch.Tensor] = None
    random_negative_x_peptide: Optional[torch.Tensor] = None
    random_negative_x_allele: Optional[torch.Tensor] = None
    combined_peptide: Optional[torch.Tensor] = None
    combined_allele: Optional[torch.Tensor] = None

    @property
    def device(self):
        """The torch device holding this row space."""
        return self.x_peptide.device

    @property
    def row_count(self):
        """Total rows in the logical ``[random negatives | real]`` space."""
        return int(self.y_encoded.shape[0])

    @property
    def random_negative_count(self):
        """Number of mutable random-negative rows at the front of the space."""
        if self.random_negative_x_peptide is None:
            return 0
        return int(self.random_negative_x_peptide.shape[0])

    @classmethod
    def from_arrays(
        cls,
        *,
        x_peptide,
        x_allele,
        y_encoded,
        sample_weights,
        random_negative_x_peptide_template,
        random_negative_x_allele,
        device,
    ):
        """Materialize a device-resident affinity training row space.

        Parameters mirror the arrays produced by ``Class1NeuralNetwork.fit``.
        ``random_negative_x_peptide_template`` provides only shape and dtype
        for the mutable random-negative slice; epoch contents are filled via
        :meth:`refill_random_negative_peptides`.

        The returned object is the only data source used by the inner fit loop:
        callers ask for batches by row index with :meth:`batch_for_indices`.
        """

        def _to_device(value, dtype=None):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value)
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            return tensor.to(device, non_blocking=False)

        def _peptide_to_device(value):
            tensor = _to_device(value)
            if (
                    tensor is not None
                    and tensor.ndim > 2
                    and not tensor.dtype.is_floating_point):
                tensor = tensor.to(torch.float32)
            return tensor

        def _peptide_template_shape_dtype(value):
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value)
            dtype = tensor.dtype
            if tensor.ndim > 2 and not dtype.is_floating_point:
                dtype = torch.float32
            return tuple(tensor.shape), dtype

        x_peptide_dev = _peptide_to_device(x_peptide)
        x_allele_dev = _to_device(x_allele)
        if x_peptide_dev is None:
            raise ValueError(
                "AffinityDeviceTrainingData.from_arrays: x_peptide is required."
            )
        num_real = int(x_peptide_dev.shape[0])
        num_random_negatives = 0

        combined_peptide = None
        rn_peptide_view = None
        x_peptide_view = x_peptide_dev
        if random_negative_x_peptide_template is not None:
            rn_shape, rn_dtype = _peptide_template_shape_dtype(
                random_negative_x_peptide_template
            )
            num_rn = int(rn_shape[0])
            num_random_negatives = num_rn
            if x_peptide_dev.dtype != rn_dtype:
                x_peptide_dev = x_peptide_dev.to(rn_dtype)
            combined_shape = (num_rn + num_real, *rn_shape[1:])
            if tuple(x_peptide_dev.shape[1:]) != tuple(rn_shape[1:]):
                raise ValueError(
                    "AffinityDeviceTrainingData.from_arrays: real and "
                    "random-negative peptide tensors disagree on per-row "
                    "shape: %r vs %r" % (
                        tuple(x_peptide_dev.shape[1:]),
                        tuple(rn_shape[1:]),
                    )
                )
            combined_peptide = torch.empty(
                combined_shape, dtype=rn_dtype, device=device
            )
            rn_peptide_view = combined_peptide[:num_rn]
            x_peptide_view = combined_peptide[num_rn:]
            x_peptide_view.copy_(x_peptide_dev)
            rn_peptide_view.zero_()

        rn_allele_dev = _to_device(random_negative_x_allele)
        combined_allele = None
        rn_allele_view = rn_allele_dev
        x_allele_view = x_allele_dev
        if x_allele_dev is not None and int(x_allele_dev.shape[0]) != num_real:
            raise ValueError(
                "AffinityDeviceTrainingData.from_arrays: x_allele has %d rows "
                "but x_peptide has %d rows." % (
                    int(x_allele_dev.shape[0]),
                    num_real,
                )
            )
        if num_random_negatives and (x_allele_dev is None) != (rn_allele_dev is None):
            raise ValueError(
                "AffinityDeviceTrainingData.from_arrays: real and "
                "random-negative allele tensors must either both be present "
                "or both be absent."
            )
        if rn_allele_dev is not None and x_allele_dev is not None:
            if int(rn_allele_dev.shape[0]) != num_random_negatives:
                raise ValueError(
                    "AffinityDeviceTrainingData.from_arrays: "
                    "random_negative_x_allele has %d rows but random-negative "
                    "peptide space has %d rows." % (
                        int(rn_allele_dev.shape[0]),
                        num_random_negatives,
                    )
                )
            if rn_allele_dev.dtype != x_allele_dev.dtype:
                target_dtype = (
                    torch.float32
                    if (rn_allele_dev.dtype.is_floating_point
                        or x_allele_dev.dtype.is_floating_point)
                    else x_allele_dev.dtype
                )
                rn_allele_dev = rn_allele_dev.to(target_dtype)
                x_allele_dev = x_allele_dev.to(target_dtype)
            num_rn_a = int(rn_allele_dev.shape[0])
            num_real_a = int(x_allele_dev.shape[0])
            combined_allele = torch.empty(
                (num_rn_a + num_real_a, *rn_allele_dev.shape[1:]),
                dtype=rn_allele_dev.dtype,
                device=device,
            )
            combined_allele[:num_rn_a].copy_(rn_allele_dev)
            combined_allele[num_rn_a:].copy_(x_allele_dev)
            rn_allele_view = combined_allele[:num_rn_a]
            x_allele_view = combined_allele[num_rn_a:]

        total_rows = num_random_negatives + num_real
        y_encoded_dev = _to_device(y_encoded, dtype=torch.float32)
        if y_encoded_dev is None:
            raise ValueError(
                "AffinityDeviceTrainingData.from_arrays: y_encoded is required."
            )
        if int(y_encoded_dev.shape[0]) != total_rows:
            raise ValueError(
                "AffinityDeviceTrainingData.from_arrays: y_encoded has %d rows "
                "but peptide row space has %d rows." % (
                    int(y_encoded_dev.shape[0]),
                    total_rows,
                )
            )

        sample_weights_dev = _to_device(sample_weights, dtype=torch.float32)
        if (
                sample_weights_dev is not None
                and int(sample_weights_dev.shape[0]) != total_rows):
            raise ValueError(
                "AffinityDeviceTrainingData.from_arrays: sample_weights has "
                "%d rows but peptide row space has %d rows." % (
                    int(sample_weights_dev.shape[0]),
                    total_rows,
                )
            )

        return cls(
            x_peptide=x_peptide_view,
            x_allele=x_allele_view,
            y_encoded=y_encoded_dev,
            sample_weights=sample_weights_dev,
            random_negative_x_peptide=rn_peptide_view,
            random_negative_x_allele=rn_allele_view,
            combined_peptide=combined_peptide,
            combined_allele=combined_allele,
        )

    def refill_random_negative_peptides(self, encoded_peptides):
        """Overwrite the mutable random-negative peptide slice in place.

        Does nothing when the fit has no random negatives. ``encoded_peptides``
        may be a numpy array, CPU tensor, or tensor already on the target device;
        dtype is coerced to the existing random-negative slice.
        """
        if self.random_negative_x_peptide is None:
            return
        source = encoded_peptides
        if not isinstance(source, torch.Tensor):
            source = torch.as_tensor(source)
        self.random_negative_x_peptide.copy_(
            source.to(
                self.random_negative_x_peptide.device,
                dtype=self.random_negative_x_peptide.dtype,
            )
        )

    def batch_for_indices(self, batch_indices):
        """Return ``(inputs, y_batch, weights_batch)`` for device indices.

        ``batch_indices`` must be a torch integer tensor on ``self.device``.
        Keeping indices on device lets the fit loop form batches with
        ``index_select`` and no per-batch host-to-device copies.
        """
        combined_peptide = self.combined_peptide
        if combined_peptide is None:
            combined_peptide = self.x_peptide
        inputs = {"peptide": combined_peptide.index_select(0, batch_indices)}
        if self.combined_allele is not None:
            inputs["allele"] = self.combined_allele.index_select(0, batch_indices)
        elif self.x_allele is not None:
            inputs["allele"] = self.x_allele.index_select(0, batch_indices)
        y_batch = self.y_encoded.index_select(0, batch_indices)
        weights_batch = None
        if self.sample_weights is not None:
            weights_batch = self.sample_weights.index_select(0, batch_indices)
        return inputs, y_batch, weights_batch

    def batch_dict_for_indices(self, indices, device=None):
        """Return a batch dict for LSUV/data-dependent initialization."""
        if device is None:
            device = self.device
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(
                numpy.asarray(indices, dtype=numpy.int64), device=device
            )
        elif indices.device != device:
            indices = indices.to(device)
        inputs, y_batch, weights_batch = self.batch_for_indices(indices)
        batch = {"peptide": inputs["peptide"], "y": y_batch}
        if "allele" in inputs:
            batch["allele"] = inputs["allele"]
        if weights_batch is not None:
            batch["weight"] = weights_batch
        return batch
