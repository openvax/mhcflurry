"""
"""
import sys
import argparse
import os
import collections

import pandas
import tqdm  # progress bar
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "peptides",
    metavar="FILE.csv",
    help="CSV of mass spec hits")
parser.add_argument(
    "reference_csv",
    metavar="FILE.csv",
    help="CSV of protein sequences")
parser.add_argument(
    "reference_index",
    metavar="FILE.fm",
    nargs="?",
    help="Legacy shellinford index over protein sequences. Retained for "
    "compatibility with existing GENERATE.sh calls; ignored by current "
    "backends.")
parser.add_argument(
    "--backend",
    choices=["auto", "ahocorasick", "sliding"],
    default="auto",
    help="Peptide lookup backend. Default: %(default)s, which uses "
    "ahocorasick-rs when installed and otherwise falls back to a pure-Python "
    "sliding-window scan.")
parser.add_argument(
    "--out",
    metavar="OUT.csv",
    help="Out file path")
parser.add_argument(
    "--flanking-length",
    metavar="N",
    type=int,
    default=15,
    help="Length of flanking sequence to include")
parser.add_argument(
    "--debug-max-rows",
    metavar="N",
    type=int,
    default=None,
    help="Max rows to process. Useful for debugging. If specified an ipdb "
    "debugging session is also opened at the end of the script")


def non_overlapping_starts(starts, peptide_length):
    """
    Match legacy ``re.finditer(peptide, seq)`` non-overlapping semantics.
    """
    result = []
    next_allowed_start = 0
    for start in sorted(set(starts)):
        if start >= next_allowed_start:
            result.append(start)
            next_allowed_start = start + peptide_length
    return result


def normalize_occurrences(occurrences):
    result = {}
    for peptide, protein_to_starts in occurrences.items():
        peptide_length = len(peptide)
        normalized = {}
        for protein_idx, starts in protein_to_starts.items():
            starts = non_overlapping_starts(starts, peptide_length)
            if starts:
                normalized[protein_idx] = starts
        if normalized:
            result[peptide] = normalized
    return result


def find_occurrences_sliding(peptides, reference_df):
    """
    Pure-Python fallback: scan each protein by observed peptide length.
    """
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
    return normalize_occurrences(occurrences)


def find_occurrences_ahocorasick(peptides, reference_df):
    """
    Use Rust-backed Aho-Corasick exact matching when available.
    """
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
    return normalize_occurrences(occurrences)


def find_occurrences(peptides, reference_df, backend):
    if backend == "auto":
        try:
            import ahocorasick_rs  # noqa: F401
            backend = "ahocorasick"
        except ImportError:
            backend = "sliding"
            print(
                "ahocorasick-rs unavailable; using sliding-window backend",
                flush=True)

    if backend == "ahocorasick":
        print("Using ahocorasick-rs backend", flush=True)
        return find_occurrences_ahocorasick(peptides, reference_df)
    if backend == "sliding":
        print("Using sliding-window backend", flush=True)
        return find_occurrences_sliding(peptides, reference_df)

    raise ValueError("Unsupported backend: %s" % backend)


def build_join_df(df, reference_df, occurrences, flanking_length, debug_max_rows):
    join_df = []
    for row in tqdm.tqdm(
            df.itertuples(),
            total=len(df),
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
                ].rjust(flanking_length, 'X')
                c_flank = reference_row.seq[
                    end_pos: (end_pos + flanking_length)
                ].ljust(flanking_length, 'X')
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


def run(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    df = pandas.read_csv(args.peptides)
    df["hit_id"] = "hit." + df.index.map('{0:07d}'.format)
    df = df.set_index("hit_id")
    print("Read peptides", df.shape, *df.columns.tolist())

    reference_df = pandas.read_csv(args.reference_csv, index_col=0)
    reference_df = reference_df.set_index("accession")
    print("Read proteins", reference_df.shape, *reference_df.columns.tolist())

    peptides = set(df.peptide)
    print("Unique peptides", len(peptides))
    occurrences = find_occurrences(peptides, reference_df, args.backend)
    print("Peptides with protein matches", len(occurrences))

    join_df = build_join_df(
        df,
        reference_df,
        occurrences,
        args.flanking_length,
        args.debug_max_rows)

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

    merged_df = pandas.merge(
        join_df,
        df,
        how="left",
        left_on="hit_id",
        right_index=True)

    merged_df.to_csv(args.out, index=False)
    print("Wrote: %s" % os.path.abspath(args.out))

    if args.debug_max_rows:
        # Leave user in a debugger
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    run()
