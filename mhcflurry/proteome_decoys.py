# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for generating proteome peptide decoy candidates."""
from __future__ import annotations

import pandas
import numpy

from .amino_acid import COMMON_AMINO_ACIDS


PROTEOME_PEPTIDE_COLUMNS = [
    "protein_accession",
    "peptide",
    "n_flank",
    "c_flank",
    "start_position",
]


def unique_in_order(values):
    """Return unique non-null values while preserving first-seen order."""
    return list(dict.fromkeys(value for value in values if pandas.notnull(value)))


def infer_flanking_length(hit_df):
    """Infer the fixed flanking sequence length used for peptide decoys."""
    lengths = set(hit_df.n_flank.dropna().str.len().unique())
    lengths.update(hit_df.c_flank.dropna().str.len().unique())
    if len(lengths) != 1:
        raise ValueError("Expected one flank length, got %s" % sorted(lengths))
    return lengths.pop()


def load_reference_sequences(reference_csv, accessions):
    """Load protein sequences keyed by accession for the requested accessions."""
    accessions = unique_in_order(accessions)
    try:
        reference_df = pandas.read_csv(
            reference_csv,
            usecols=["accession", "seq"])
    except ValueError:
        reference_df = pandas.read_csv(reference_csv)
    if "accession" not in reference_df.columns:
        raise ValueError(
            "Expected reference CSV %s to have an 'accession' column" %
            reference_csv)
    if "seq" not in reference_df.columns:
        raise ValueError(
            "Expected reference CSV %s to have a 'seq' column" % reference_csv)

    reference_df = (
        reference_df
        .drop_duplicates("accession")
        .set_index("accession")
    )
    missing = [accession for accession in accessions
               if accession not in reference_df.index]
    if missing:
        raise ValueError(
            "Missing %d protein accessions in %s, including: %s" % (
                len(missing), reference_csv, ", ".join(missing[:10])))
    return reference_df.loc[accessions].seq.to_dict()


def iter_protein_peptide_records(
        accession,
        sequence,
        lengths=(8, 9, 10, 11),
        flanking_length=15,
        valid_amino_acids=None):
    """Yield peptide/flank records for one protein sequence.

    The start-position range intentionally matches the historical
    ``write_proteome_peptides.py`` behavior so release data generation remains
    comparable. Concretely, ``range(0, len(sequence) - min_length)`` has an
    exclusive upper bound, so the last start position is
    ``len(sequence) - min_length - 1`` and the single ``min_length``-mer that
    would start at ``len(sequence) - min_length`` (the C-terminal
    ``min_length``-mer) is NOT emitted. This off-by-one is deliberate
    bug-for-bug parity, not an oversight: changing it would alter the decoy
    set used to generate released models. ``test_iter_protein_peptide_records_*``
    pins the resulting counts so the behavior can't drift silently.
    """
    valid_amino_acids = set(valid_amino_acids or COMMON_AMINO_ACIDS)
    lengths = sorted(lengths)
    min_length = min(lengths)
    for start in range(0, len(sequence) - min_length):
        for length in lengths:
            end_pos = start + length
            if end_pos > len(sequence):
                break
            peptide = sequence[start:end_pos]
            if any(letter not in valid_amino_acids for letter in peptide):
                continue
            n_flank = sequence[
                max(start - flanking_length, 0):start
            ].rjust(flanking_length, "X")
            c_flank = sequence[
                end_pos:(end_pos + flanking_length)
            ].ljust(flanking_length, "X")
            yield accession, peptide, n_flank, c_flank, start


def make_peptide_frame_for_accessions(
        accessions,
        sequences_by_accession,
        lengths=(8, 9, 10, 11),
        flanking_length=15,
        valid_amino_acids=None):
    """Generate a peptide/flank DataFrame for the requested accessions."""
    rows = []
    for accession in unique_in_order(accessions):
        rows.extend(iter_protein_peptide_records(
            accession=accession,
            sequence=sequences_by_accession[accession],
            lengths=lengths,
            flanking_length=flanking_length,
            valid_amino_acids=valid_amino_acids,
        ))
    return pandas.DataFrame.from_records(rows, columns=PROTEOME_PEPTIDE_COLUMNS)


def sample_peptide_frame_for_accessions(
        accessions,
        sequences_by_accession,
        lengths=(8, 9, 10, 11),
        flanking_length=15,
        exclude_peptides=(),
        n=1,
        valid_amino_acids=None,
        sampling_method="positions",
        allow_smaller=False):
    """Sample windows without replacement; build flanks only for retained rows.

    Position sampling preserves the eligible population, not the historical
    reservoir's RNG sequence. Use ``sampling_method='reservoir'`` for replay.
    ``allow_smaller`` returns the complete eligible population on exhaustion.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return pandas.DataFrame(columns=PROTEOME_PEPTIDE_COLUMNS)

    if sampling_method == "positions":
        return _sample_numeric_positions(
            accessions, sequences_by_accession, lengths, flanking_length,
            exclude_peptides, n, valid_amino_acids, allow_smaller)
    if sampling_method != "reservoir":
        raise ValueError("Unknown peptide sampling method: " + sampling_method)

    exclude_peptides = set(exclude_peptides)
    reservoir = []
    seen = 0
    for accession in unique_in_order(accessions):
        for record in iter_protein_peptide_records(
                accession=accession,
                sequence=sequences_by_accession[accession],
                lengths=lengths,
                flanking_length=flanking_length,
                valid_amino_acids=valid_amino_acids):
            if record[1] in exclude_peptides:
                continue
            seen += 1
            if len(reservoir) < n:
                reservoir.append(record)
                continue
            replace_index = numpy.random.randint(seen)
            if replace_index < n:
                reservoir[replace_index] = record

    if seen < n and not allow_smaller:
        raise ValueError(
            "Cannot take a larger sample than population when "
            "'replace=False' (requested %d, population %d)" % (n, seen))
    return pandas.DataFrame.from_records(
        reservoir,
        columns=PROTEOME_PEPTIDE_COLUMNS)


def _sample_numeric_positions(accessions, sequences, lengths, flank_length,
                              excluded, n, valid_amino_acids, allow_smaller):
    """Draw from the same valid window population as the reference iterator."""
    lengths = sorted(lengths)
    if not lengths or lengths[0] <= 0:
        raise ValueError("Peptide lengths must be positive")
    alphabet = set(valid_amino_acids or COMMON_AMINO_ACIDS)
    excluded = set(excluded)
    segments = []
    counts = []
    for accession in unique_in_order(accessions):
        sequence = sequences[accession]
        invalid = numpy.fromiter((c not in alphabet for c in sequence), dtype="int8")
        prefix = numpy.concatenate(([0], numpy.cumsum(invalid)))
        for length in lengths:
            # Preserve the iterator's exclusive minimum-length terminal bound.
            count = max(0, min(len(sequence) - lengths[0], len(sequence) - length + 1))
            starts = numpy.arange(count)
            starts = starts[(prefix[starts + length] - prefix[starts]) == 0]
            if len(starts):
                segments.append((accession, sequence, length, starts))
                counts.append(len(starts))
    ends = numpy.cumsum(counts, dtype="int64")
    total = int(ends[-1]) if len(ends) else 0
    # Only numeric indices are shuffled. No peptide/flank tuples are created
    # for the unsampled population. Rejecting excluded peptides from a random
    # permutation is uniform sampling of the remaining eligible windows.
    positions = numpy.random.permutation(total)
    rows = []
    for offset in range(0, total, 65536):
        chunk = positions[offset:offset + 65536]
        segment_indices = numpy.searchsorted(ends, chunk, side="right")
        for index, segment_index in zip(chunk, segment_indices):
            accession, sequence, length, starts = segments[segment_index]
            base = int(ends[segment_index - 1]) if segment_index else 0
            start = int(starts[index - base])
            peptide = sequence[start:start + length]
            if peptide in excluded:
                continue
            end = start + length
            rows.append((accession, peptide,
                         sequence[max(0, start - flank_length):start].rjust(flank_length, "X"),
                         sequence[end:end + flank_length].ljust(flank_length, "X"), start))
            if len(rows) == n:
                return pandas.DataFrame.from_records(rows, columns=PROTEOME_PEPTIDE_COLUMNS)
    if not allow_smaller:
        raise ValueError("Cannot take a larger sample than population when 'replace=False' "
                         "(requested %d, population %d)" % (n, len(rows)))
    return pandas.DataFrame.from_records(rows, columns=PROTEOME_PEPTIDE_COLUMNS)


def peptides_by_length_from_frame(peptide_df):
    """Return peptide DataFrames keyed by peptide length."""
    peptide_df = peptide_df.copy()
    peptide_df["length"] = peptide_df.peptide.str.len()
    return dict(iter(peptide_df.groupby("length")))
