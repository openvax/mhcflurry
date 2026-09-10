"""String-free peptide inputs with the existing class-I alignment semantics."""

import numpy
import torch

from .amino_acid import AMINO_ACID_INDEX
from .encodable_sequences import EncodingError


UNKNOWN_INDEX = AMINO_ACID_INDEX["X"]


class NumericSequences:
    """Unaligned integer residue rows and actual lengths, on CPU or accelerator.

    Parameters
    ----------
    indices : array-like or torch.Tensor
        Shape (N, width); entries index ``AMINO_ACID_INDEX``. Padding is ignored.
    lengths : array-like or torch.Tensor
        N integer lengths between zero and width.
    """

    def __init__(self, indices, lengths):
        indices = torch.as_tensor(indices)
        lengths = torch.as_tensor(lengths, device=indices.device)
        integers = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if indices.ndim != 2 or lengths.ndim != 1 or len(indices) != len(lengths):
            raise ValueError("Numeric peptides require an (N, width) matrix and N lengths")
        if indices.dtype not in integers or lengths.dtype not in integers:
            raise ValueError("Numeric peptide indices and lengths must be integers")
        if ((lengths < 0) | (lengths > indices.shape[1])).any():
            raise ValueError("Numeric peptide length exceeds its row width")
        valid = torch.arange(indices.shape[1], device=indices.device)[None, :] < lengths[:, None]
        if (valid & ((indices < 0) | (indices > UNKNOWN_INDEX))).any():
            raise ValueError("Invalid numeric amino-acid index")
        self.indices = torch.where(valid, indices, UNKNOWN_INDEX).to(torch.int8)
        self.lengths = lengths.to(torch.int64).clone()

    def __len__(self):
        return len(self.lengths)

    @classmethod
    def from_strings(cls, sequences, width=None):
        """Encode existing strings once, with vectorized ASCII lookup."""
        sequences = list(sequences)
        if not all(isinstance(value, str) for value in sequences):
            raise ValueError("Sequence of strings is required")
        lengths = numpy.fromiter(map(len, sequences), dtype="int64", count=len(sequences))
        width = int(lengths.max()) if width is None and len(lengths) else (width or 0)
        if len(lengths) and lengths.max() > width:
            raise ValueError("Sequence length exceeds numeric width")
        if not width:
            return cls(numpy.empty((len(sequences), 0), dtype="int8"), lengths)
        lookup = numpy.full(256, -1, dtype="int8")
        for letter, index in AMINO_ACID_INDEX.items():
            lookup[ord(letter)] = index
        encoded = numpy.asarray(sequences, dtype="S%d" % width).view("uint8").reshape(-1, width)
        valid = numpy.arange(width)[None, :] < lengths[:, None]
        return cls(numpy.where(valid, lookup[encoded], UNKNOWN_INDEX), lengths)

    def take(self, rows):
        """Select a vector of integer row indices or a same-length boolean mask."""
        rows = torch.as_tensor(rows, device=self.indices.device)
        if rows.ndim != 1:
            raise ValueError("Numeric row selection must be one-dimensional")
        if rows.dtype == torch.bool:
            if len(rows) != len(self):
                raise ValueError("Numeric row mask must match sequence count")
        elif rows.numel() and rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Numeric row indices must be integers")
        else:
            rows = rows.to(torch.long)
        return type(self)(self.indices[rows], self.lengths[rows])

    def to_strings(self):
        """Materialize strings for export or diagnostics, never for scoring."""
        alphabet = numpy.empty(len(AMINO_ACID_INDEX), dtype="S1")
        for letter, index in AMINO_ACID_INDEX.items():
            alphabet[index] = letter.upper().encode("ascii")
        rows = alphabet[self.indices.cpu().numpy()]
        return numpy.asarray([row.tobytes()[:int(length)].decode("ascii")
                              for row, length in zip(rows, self.lengths.cpu().numpy())])

    def aligned_tensor(self, peptide_encoding, device=None):
        """Gather the exact categorical network input on the requested device."""
        method = peptide_encoding.get("alignment_method", "pad_middle")
        width = int(peptide_encoding.get("max_length", 15))
        left = int(peptide_encoding.get("left_edge", 4))
        right = int(peptide_encoding.get("right_edge", 4))
        trim = peptide_encoding.get("trim", False)
        if width < 1 or left < 0 or right < 0:
            raise ValueError("Invalid peptide alignment widths")
        if trim and method not in ("left_pad", "right_pad"):
            raise NotImplementedError("trim not supported")
        minimum = left + right if method == "pad_middle" else (
            1 if method in ("left_pad", "right_pad") else 5)
        if ((self.lengths < minimum) | ((self.lengths > width) & (not trim))).any():
            raise EncodingError("Numeric peptide lengths outside supported range",
                                supported_peptide_lengths=(minimum, width))
        data = self.indices.to(device=device or self.indices.device)
        lengths = self.lengths.to(device=data.device)[:, None]
        positions = torch.arange(width, device=data.device)[None, :].expand(len(self), -1)
        if method == "pad_middle":
            middle_start = left + (width - lengths + 1) // 2
            source = torch.where(positions < left, positions,
                torch.where(positions >= width - right, lengths - right + positions - (width - right),
                            left + positions - middle_start))
            valid = ((positions < left) | (positions >= width - right) |
                     ((positions >= middle_start) & (positions < middle_start + lengths - left - right)))
        elif method in ("left_pad", "right_pad", "left_pad_right_pad", "left_pad_centered_right_pad"):
            left_aligned = positions
            right_aligned = positions - (width - lengths)
            if method == "right_pad":
                source = left_aligned
            elif method == "left_pad":
                source = right_aligned
            elif method == "left_pad_right_pad":
                source = torch.cat([left_aligned, right_aligned], dim=1)
            else:
                centered = positions - (width - lengths) // 2
                source = torch.cat([left_aligned, centered, right_aligned], dim=1)
            valid = (source >= 0) & (source < lengths)
        else:
            raise NotImplementedError("Unsupported alignment method: " + method)
        if not len(self):
            return torch.empty(source.shape, dtype=torch.int8, device=data.device)
        return torch.where(valid, data.gather(1, source.clamp(0, data.shape[1] - 1)), UNKNOWN_INDEX)
