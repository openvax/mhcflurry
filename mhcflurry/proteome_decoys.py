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
    comparable.
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
        valid_amino_acids=None):
    """Uniformly sample peptide/flank records without materializing candidates."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return pandas.DataFrame(columns=PROTEOME_PEPTIDE_COLUMNS)

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

    if seen < n:
        raise ValueError(
            "Cannot take a larger sample than population when "
            "'replace=False' (requested %d, population %d)" % (n, seen))
    return pandas.DataFrame.from_records(
        reservoir,
        columns=PROTEOME_PEPTIDE_COLUMNS)


def peptides_by_length_from_frame(peptide_df):
    """Return peptide DataFrames keyed by peptide length."""
    peptide_df = peptide_df.copy()
    peptide_df["length"] = peptide_df.peptide.str.len()
    return dict(iter(peptide_df.groupby("length")))
