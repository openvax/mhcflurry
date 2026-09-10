"""Reusable numeric processing candidates; strings are an export operation."""

from dataclasses import dataclass

import numpy
import pandas
import torch

from .amino_acid import AMINO_ACID_INDEX, COMMON_AMINO_ACIDS
from .numeric_sequences import NumericSequences, UNKNOWN_INDEX
from .proteome_decoys import PROTEOME_PEPTIDE_COLUMNS, unique_in_order


MAX_PROCESSING_LENGTH = 11


def sequence_keys(indices):
    """Collision-free fixed-width numeric identity, including X padding."""
    indices = numpy.ascontiguousarray(indices, dtype="int8")
    if indices.ndim != 2 or indices.shape[1] != MAX_PROCESSING_LENGTH:
        raise ValueError("Processing identities require eleven numeric columns")
    return indices.view("V%d" % MAX_PROCESSING_LENGTH).ravel()


class ProcessingProteome:
    """Encode source proteins once; sample numeric positions and gather on device.

    CPU sampling preserves the historical eligible-window population and RNG
    stream when passed an equivalent ``numpy.random.RandomState``. A device copy
    is created lazily by the scoring owner, never by CPU preparation threads.
    """

    def __init__(self, sequences):
        self.accessions = list(sequences)
        self.sequences = list(sequences.values())
        self.accession_index = {name: i for i, name in enumerate(self.accessions)}
        self.lengths = numpy.asarray([len(s) for s in self.sequences], dtype="int64")
        self.offsets = numpy.r_[0, numpy.cumsum(self.lengths)]
        lookup = numpy.full(256, -1, dtype="int8")
        for letter in COMMON_AMINO_ACIDS:
            lookup[ord(letter)] = AMINO_ACID_INDEX[letter]
        encoded = [lookup[numpy.frombuffer(s.encode("ascii", errors="replace"), dtype="uint8")]
                   for s in self.sequences]
        self.residues = numpy.concatenate(encoded) if encoded else numpy.empty(0, dtype="int8")
        self.invalid_prefix = [numpy.r_[0, numpy.cumsum(row < 0)] for row in encoded]
        self._devices = {}

    def raw_rows(self, proteins, starts, lengths):
        """Gather CPU numeric identities for filtering without string creation."""
        if not len(starts):
            return numpy.empty((0, MAX_PROCESSING_LENGTH), dtype="int8")
        columns = numpy.arange(MAX_PROCESSING_LENGTH)
        positions = self.offsets[proteins, None] + starts[:, None] + columns
        return numpy.where(columns < lengths[:, None],
            self.residues[positions.clip(0, len(self.residues) - 1)], UNKNOWN_INDEX)

    def gather(self, windows, device="cpu"):
        """Gather selected peptide rows into Torch directly from encoded proteins."""
        if windows.proteome is not self:
            raise ValueError("Windows belong to a different encoded proteome")
        device = torch.device(device)
        key = str(device)
        if key not in self._devices:
            self._devices[key] = torch.as_tensor(self.residues, device=device)
        if not len(windows):
            return NumericSequences(torch.empty((0, MAX_PROCESSING_LENGTH), dtype=torch.int8, device=device),
                                    torch.empty(0, dtype=torch.long, device=device))
        starts = torch.as_tensor(self.offsets[windows.proteins] + windows.starts, device=device)
        lengths = torch.as_tensor(windows.lengths, device=device)
        columns = torch.arange(MAX_PROCESSING_LENGTH, device=device)
        positions = starts[:, None] + columns
        rows = self._devices[key][positions.clamp(0, len(self.residues) - 1)]
        return NumericSequences(torch.where(columns < lengths[:, None], rows, UNKNOWN_INDEX), lengths)

    def sample(self, accessions, length, n, excluded, rng):
        """Draw n eligible windows without replacement, then deduplicate peptides.

        Exhaustion returns the available population. As in the previous caller,
        duplicate sequences are removed *after* n position draws, keeping the
        first retained source position. No peptide/flank strings are constructed.
        """
        if not 1 <= length <= MAX_PROCESSING_LENGTH or n < 0:
            raise ValueError("Invalid processing window length or count")
        if n == 0:
            empty = numpy.empty(0, dtype="int64")
            return ProteinWindows(self, empty, empty, empty)
        starts, proteins = [], []
        for accession in unique_in_order(accessions):
            i = self.accession_index[accession]
            # Preserve the historical excluded final length-mer start.
            candidates = numpy.arange(max(0, self.lengths[i] - length))
            prefix = self.invalid_prefix[i]
            candidates = candidates[(prefix[candidates + length] - prefix[candidates]) == 0]
            starts.append(candidates)
            proteins.append(numpy.full(len(candidates), i, dtype="int64"))
        starts = numpy.concatenate(starts) if starts else numpy.empty(0, dtype="int64")
        proteins = numpy.concatenate(proteins) if proteins else numpy.empty(0, dtype="int64")
        excluded = [p for p in excluded if len(p) == length]
        excluded_keys = sequence_keys(NumericSequences.from_strings(excluded, MAX_PROCESSING_LENGTH).indices.numpy())
        order = rng.permutation(len(starts))
        retained = []
        remaining = n
        for offset in range(0, len(order), 65536):
            if not remaining:
                break
            draw = order[offset:offset + 65536]
            lengths = numpy.full(len(draw), length, dtype="int64")
            keys = sequence_keys(self.raw_rows(proteins[draw], starts[draw], lengths))
            keep = draw[~numpy.isin(keys, excluded_keys)][:remaining]
            retained.append(keep)
            remaining -= len(keep)
        retained = numpy.concatenate(retained) if retained else numpy.empty(0, dtype="int64")
        result = ProteinWindows(self, proteins[retained], starts[retained], numpy.full(len(retained), length, dtype="int64"))
        _, first = numpy.unique(result.keys, return_index=True)
        return result.take(numpy.sort(first))


@dataclass
class ProteinWindows:
    """Selected source positions with collision-free numeric peptide identities.

    ``proteins``, ``starts`` and ``lengths`` are equally sized one-dimensional
    integer arrays. Starts are zero-based; each window must fit its protein and
    have a length between one and ``MAX_PROCESSING_LENGTH``.
    """

    proteome: ProcessingProteome
    proteins: numpy.ndarray
    starts: numpy.ndarray
    lengths: numpy.ndarray

    def __post_init__(self):
        for name in ("proteins", "starts", "lengths"):
            values = numpy.asarray(getattr(self, name))
            if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
                raise ValueError("Window positions and lengths must be integer vectors")
            setattr(self, name, values.astype("int64", copy=True))
        if not (len(self.proteins) == len(self.starts) == len(self.lengths)):
            raise ValueError("Window position vectors have different lengths")
        if ((self.proteins < 0) | (self.proteins >= len(self.proteome.lengths))).any():
            raise ValueError("Window refers to an unknown protein index")
        if ((self.starts < 0) | (self.lengths < 1) |
                (self.lengths > MAX_PROCESSING_LENGTH) |
                (self.starts > self.proteome.lengths[self.proteins] - self.lengths)).any():
            raise ValueError("Window lies outside its protein or supported peptide lengths")

    def __len__(self):
        return len(self.starts)

    @property
    def keys(self):
        return sequence_keys(self.proteome.raw_rows(self.proteins, self.starts, self.lengths))

    def take(self, rows):
        return type(self)(self.proteome, self.proteins[rows], self.starts[rows], self.lengths[rows])

    @classmethod
    def concatenate(cls, parts):
        if not parts or any(part.proteome is not parts[0].proteome for part in parts):
            raise ValueError("Window batches must share one encoded proteome")
        return cls(parts[0].proteome, *(numpy.concatenate([getattr(p, attr) for p in parts])
                                       for attr in ("proteins", "starts", "lengths")))

    def to_frame(self, flank_length):
        """Construct sequence/flank strings only when exporting scored artifacts."""
        if not isinstance(flank_length, (int, numpy.integer)) or flank_length < 0:
            raise ValueError("Flank length must be a nonnegative integer")
        rows = []
        for protein, start, length in zip(self.proteins, self.starts, self.lengths):
            sequence = self.proteome.sequences[protein]
            end = start + length
            rows.append((self.proteome.accessions[protein], sequence[start:end],
                         sequence[max(0, start - flank_length):start].rjust(flank_length, "X"),
                         sequence[end:end + flank_length].ljust(flank_length, "X"), start))
        return pandas.DataFrame.from_records(rows, columns=PROTEOME_PEPTIDE_COLUMNS)


class NumericCandidatePool:
    """Observed rows plus numeric negatives, ready for scoring then export."""

    def __init__(self, hits, windows, sample, flank_length):
        self.hits = hits
        self.windows = windows
        self.sample = sample
        self.flank_length = flank_length
        self.hit_peptides = NumericSequences.from_strings(hits.peptide, MAX_PROCESSING_LENGTH)
        keys = numpy.r_[sequence_keys(self.hit_peptides.indices.numpy()), windows.keys]
        _, first, inverse = numpy.unique(keys, return_index=True, return_inverse=True)
        order = numpy.argsort(first)
        self.unique_rows = first[order]
        self.inverse = numpy.argsort(order)[inverse]

    def prediction_input(self, device):
        """Use the scoring owner's device; no strings or host round trip."""
        candidates = self.windows.proteome.gather(self.windows, device)
        return NumericSequences(torch.cat([self.hit_peptides.indices.to(device), candidates.indices]),
            torch.cat([self.hit_peptides.lengths.to(device), candidates.lengths])).take(self.unique_rows)

    def with_scores(self, values):
        """Export a scored pool in the established observed-then-decoy order."""
        values = numpy.asarray(values)
        if values.shape != (len(self.unique_rows),):
            raise ValueError("Numeric candidate score count mismatch")
        negatives = self.windows.to_frame(self.flank_length).drop(columns="start_position")
        negatives["hit"] = 0
        frame = pandas.concat([self.hits, negatives], ignore_index=True, sort=False)
        frame["sample_id"] = self.sample
        frame["affinity_prediction"] = values[self.inverse]
        return frame
