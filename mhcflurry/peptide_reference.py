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

"""Helpers for matching observed peptides to reference proteins."""

import collections

import pandas
import tqdm

tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


def normalize_reference_dataframe(reference_df):
    """Return reference proteins indexed by accession."""
    if "accession" not in reference_df.columns:
        if reference_df.index.name == "accession":
            return reference_df
        raise ValueError("Expected reference_df to include an accession column")
    return reference_df.set_index("accession")


def annotate_peptide_references(
        peptide_df,
        reference_df,
        flanking_length=15,
        debug_max_rows=None):
    """Annotate peptides with matching reference proteins and flanks.

    Parameters
    ----------
    peptide_df : pandas.DataFrame
        DataFrame with a ``peptide`` column. Its index is carried into the
        output as ``hit_id``.
    reference_df : pandas.DataFrame
        DataFrame with protein ``accession`` and ``seq`` columns.
    flanking_length : int
        Number of residues to include on each side of the peptide.
    debug_max_rows : int, optional
        Stop building annotations after this many reference rows have been
        produced. Intended only for generation-script debugging.

    Returns
    -------
    pandas.DataFrame
        Original peptide rows joined to their reference protein annotations.
    """
    return _annotate_peptide_references(
        peptide_df,
        reference_df,
        flanking_length=flanking_length,
        debug_max_rows=debug_max_rows,
        backend="auto")


def _annotate_peptide_references(
        peptide_df,
        reference_df,
        flanking_length=15,
        debug_max_rows=None,
        backend="auto"):
    reference_df = normalize_reference_dataframe(reference_df)
    if "seq" not in reference_df.columns:
        raise ValueError("Expected reference_df to include a seq column")
    if "peptide" not in peptide_df.columns:
        raise ValueError("Expected peptide_df to include a peptide column")

    peptides = set(peptide_df.peptide)
    print("Unique peptides", len(peptides))
    occurrences = _find_peptide_occurrences(
        peptides,
        reference_df,
        backend=backend)
    print("Peptides with protein matches", len(occurrences))

    join_df = _build_join_dataframe(
        peptide_df,
        reference_df,
        occurrences,
        flanking_length,
        debug_max_rows)

    join_df["protein_accession"] = join_df.match_index.map(
        reference_df.index.to_series().reset_index(drop=True))

    del join_df["match_index"]

    protein_cols = [
        c for c in reference_df.columns
        if c not in ["name", "description", "seq"]
    ]
    for col in protein_cols:
        join_df["protein_%s" % col] = join_df.protein_accession.map(
            reference_df[col])

    return pandas.merge(
        join_df,
        peptide_df,
        how="left",
        left_on="hit_id",
        right_index=True)


def _non_overlapping_starts(starts, peptide_length):
    """Match legacy ``re.finditer(peptide, seq)`` semantics."""
    result = []
    next_allowed_start = 0
    for start in sorted(set(starts)):
        if start >= next_allowed_start:
            result.append(start)
            next_allowed_start = start + peptide_length
    return result


def _normalize_occurrences(occurrences):
    result = {}
    for peptide, protein_to_starts in occurrences.items():
        peptide_length = len(peptide)
        normalized = {}
        for protein_idx, starts in protein_to_starts.items():
            starts = _non_overlapping_starts(starts, peptide_length)
            if starts:
                normalized[protein_idx] = starts
        if normalized:
            result[peptide] = normalized
    return result


def _find_occurrences_sliding(peptides, reference_df):
    """Pure-Python fallback: scan each protein by observed peptide length."""
    peptides_by_length = collections.defaultdict(set)
    for peptide in peptides:
        peptides_by_length[len(peptide)].add(peptide)
    length_sets = sorted(peptides_by_length.items())

    occurrences = collections.defaultdict(lambda: collections.defaultdict(list))
    for protein_idx, seq in tqdm.tqdm(
            enumerate(reference_df.seq),
            total=len(reference_df),
            desc="Scanning proteins"):
        seq_len = len(seq)
        for length, length_peptides in length_sets:
            stop = seq_len - length + 1
            if stop <= 0:
                continue
            for start in range(stop):
                peptide = seq[start:start + length]
                if peptide in length_peptides:
                    occurrences[peptide][protein_idx].append(start)
    return _normalize_occurrences(occurrences)


def _find_occurrences_ahocorasick(peptides, reference_df):
    """Use Rust-backed Aho-Corasick exact matching."""
    import ahocorasick_rs

    peptides = sorted(peptides)
    automaton = ahocorasick_rs.AhoCorasick(
        peptides,
        matchkind=ahocorasick_rs.MATCHKIND_STANDARD)

    occurrences = collections.defaultdict(lambda: collections.defaultdict(list))
    for protein_idx, seq in tqdm.tqdm(
            enumerate(reference_df.seq),
            total=len(reference_df),
            desc="Scanning proteins"):
        for pattern_idx, start, end in automaton.find_matches_as_indexes(
                seq,
                overlapping=True):
            peptide = peptides[pattern_idx]
            if end - start == len(peptide):
                occurrences[peptide][protein_idx].append(start)
    return _normalize_occurrences(occurrences)


def _find_peptide_occurrences(peptides, reference_df, backend="auto"):
    if backend == "auto":
        backend = "ahocorasick"

    if backend == "ahocorasick":
        print("Using ahocorasick-rs peptide reference lookup", flush=True)
        return _find_occurrences_ahocorasick(peptides, reference_df)
    if backend == "sliding":
        print("Using sliding-window peptide reference lookup", flush=True)
        return _find_occurrences_sliding(peptides, reference_df)

    raise ValueError("Unsupported peptide reference backend: %s" % backend)


def _build_join_dataframe(
        peptide_df,
        reference_df,
        occurrences,
        flanking_length,
        debug_max_rows):
    join_df = []
    for row in tqdm.tqdm(
            peptide_df.itertuples(),
            total=len(peptide_df),
            desc="Building annotations"):
        hit_id = row.Index
        peptide = row.peptide
        protein_to_starts = occurrences.get(peptide, {})
        for match_index, starts in sorted(protein_to_starts.items()):
            reference_row = reference_df.iloc[match_index]
            for start in starts:
                end_pos = start + len(peptide)
                n_flank = reference_row.seq[
                    max(start - flanking_length, 0): start
                ].rjust(flanking_length, "X")
                c_flank = reference_row.seq[
                    end_pos: (end_pos + flanking_length)
                ].ljust(flanking_length, "X")
                join_df.append((
                    hit_id,
                    match_index,
                    len(protein_to_starts),
                    len(starts),
                    start,
                    start / len(reference_row.seq),
                    n_flank,
                    c_flank,
                ))

        if debug_max_rows and len(join_df) > debug_max_rows:
            break

    return pandas.DataFrame(
        join_df,
        columns=[
            "hit_id",
            "match_index",
            "num_proteins",
            "num_occurrences_in_protein",
            "start_position",
            "start_fraction_in_protein",
            "n_flank",
            "c_flank",
        ]).drop_duplicates()
