"""
Generate allele sequences for pan-class I models.

Additional dependency: biopython
"""
from __future__ import print_function

import sys
import argparse

import numpy
import pandas

import mhcnames

import Bio.SeqIO  # pylint: disable=import-error


def normalize_simple(s):
    return mhcnames.normalize_allele_name(s)


def normalize_complex(s, disallowed=["MIC", "HFE"]):
    if any(item in s for item in disallowed):
        return None
    try:
        return normalize_simple(s)
    except:
        while s:
            s = ":".join(s.split(":")[:-1])
            try:
                return normalize_simple(s)
            except:
                pass
        return None


parser = argparse.ArgumentParser(usage=__doc__)

parser.add_argument(
    "aligned_fasta",
    help="Aligned sequences")

parser.add_argument(
    "--recapitulate-sequences",
    required=True,
    help="CSV giving sequences to recapitulate")

parser.add_argument(
    "--differentiate-alleles",
    help="File listing alleles to differentiate using additional positions")

parser.add_argument(
    "--out-csv",
    help="Result file")


def run():
    args = parser.parse_args(sys.argv[1:])
    print(args)

    allele_to_sequence = {}
    reader = Bio.SeqIO.parse(args.aligned_fasta, "fasta")
    for record in reader:
        name = record.description.split()[1]
        print(record.name, record.description)
        allele_to_sequence[name] = str(record.seq)

    print("Read %d aligned sequences" % len(allele_to_sequence))

    allele_sequences = pandas.Series(allele_to_sequence).to_frame()
    allele_sequences.columns = ['aligned']
    allele_sequences['aligned'] = allele_sequences['aligned'].str.replace(
        "-", "X")

    allele_sequences['normalized_allele'] = allele_sequences.index.map(normalize_complex)
    allele_sequences = allele_sequences.set_index("normalized_allele", drop=True)

    selected_positions = []

    recapitulate_df = pandas.read_csv(args.recapitulate_sequences)
    recapitulate_df["normalized_allele"] = recapitulate_df.allele.map(
        normalize_complex)
    recapitulate_df = (
        recapitulate_df
            .dropna()
            .drop_duplicates("normalized_allele")
            .set_index("normalized_allele", drop=True))

    allele_sequences["recapitulate_target"] = recapitulate_df.iloc[:,-1]

    print("Sequences in recapitulate CSV that are not in aligned fasta:")
    print(recapitulate_df.index[
        ~recapitulate_df.index.isin(allele_sequences.index)
    ].tolist())

    allele_sequences_with_target = allele_sequences.loc[
        ~allele_sequences.recapitulate_target.isnull()
    ]

    position_identities = []
    target_length = int(
        allele_sequences_with_target.recapitulate_target.str.len().max())
    for i in range(target_length):
        series_i = allele_sequences_with_target.recapitulate_target.str.get(i)
        row = []
        full_length_sequence_length = int(
            allele_sequences_with_target.aligned.str.len().max())
        for k in range(full_length_sequence_length):
            series_k = allele_sequences_with_target.aligned.str.get(k)
            row.append((series_i == series_k).mean())
        position_identities.append(row)

    position_identities = pandas.DataFrame(numpy.array(position_identities))
    selected_positions = position_identities.idxmax(1).tolist()
    fractions = position_identities.max(1)
    print("Selected positions: ", *selected_positions)
    print("Lowest concordance fraction: %0.5f" % fractions.min())
    assert fractions.min() > 0.99

    allele_sequences["recapitulated"] = allele_sequences.aligned.map(
        lambda s: "".join(s[p] for p in selected_positions))

    allele_sequences_with_target = allele_sequences.loc[
        ~allele_sequences.recapitulate_target.isnull()
    ]

    agreement = (
        allele_sequences_with_target.recapitulated ==
        allele_sequences_with_target.recapitulate_target).mean()

    print("Overall agreement: %0.5f" % agreement)
    assert agreement > 0.9

    # Add additional positions
    additional_positions = []
    if args.differentiate_alleles:
        differentiate_alleles = pandas.read_csv(
            args.differentiate_alleles).iloc[:,0].values
        print(
            "Read %d alleles to differentiate:" % len(differentiate_alleles),
            differentiate_alleles)

        to_differentiate = allele_sequences.loc[
            allele_sequences.index.isin(differentiate_alleles)
        ].copy()
        print(to_differentiate.shape)

        additional_positions = []

        # Greedy search, looking ahead 3 positions at a time.
        possible_additional_positions = set()
        for (_, sub_df) in to_differentiate.groupby("recapitulated"):
            if sub_df.aligned.nunique() > 1:
                differing = pandas.DataFrame(
                    dict([(pos, chars) for (pos, chars) in
                    enumerate(zip(*sub_df.aligned.values)) if
                    any(c != chars[0] for c in chars) and "X" not in chars])).T
                possible_additional_positions.update(differing.index.values)

        def disambiguation_score(sequences):
            counts = pandas.Series(sequences, copy=False).value_counts()
            score = -1 * (counts[counts > 1] - 1).sum()
            return score

        possible_additional_positions = sorted(possible_additional_positions)
        current_sequences = to_differentiate.recapitulated
        while current_sequences.value_counts().max() > 1:
            to_differentiate["equivalence_class_size"] = (
                current_sequences.map(current_sequences.value_counts())
            )
            print("Ambiguous alleles", " ".join(
                to_differentiate.loc[
                    to_differentiate.equivalence_class_size > 1
                ].index))
            position1s = []
            position2s = []
            position3s = []
            negative_position1_distances = []
            possible_additional_positions_scores = []
            position1_scores = []
            for position1 in possible_additional_positions:
                new_sequence1 = (
                        current_sequences +
                        to_differentiate.aligned.str.get(position1))
                negative_position1_distance = -1 * min(
                    abs(position1 - selected) for selected in selected_positions)
                position1_score = disambiguation_score(new_sequence1)

                for (i, position2) in enumerate(possible_additional_positions):
                    new_sequence2 = (
                        new_sequence1 +
                        to_differentiate.aligned.str.get(position2))
                    for position3 in possible_additional_positions:
                        new_sequence3 = (
                            new_sequence2 +
                            to_differentiate.aligned.str.get(position3))

                        score = disambiguation_score(new_sequence3)
                        position1s.append(position1)
                        position2s.append(position2)
                        position3s.append(position3)
                        possible_additional_positions_scores.append(score)
                        negative_position1_distances.append(
                            negative_position1_distance)
                        position1_scores.append(position1_score)

            scores_df = pandas.DataFrame({
                "position1": position1s,
                "position2": position2s,
                "position3": position3s,
                "negative_position1_distance": negative_position1_distances,
                "tuple_score": possible_additional_positions_scores,
                "position1_score": position1_scores,
            }).sort_values(
                ["tuple_score", "position1_score", "negative_position1_distance"],
                ascending=False)
            print(scores_df)
            selected_additional_position = scores_df.iloc[0].position1
            print("Selected additional position", selected_additional_position)
            additional_positions.append(selected_additional_position)
            current_sequences = (
                    current_sequences +
                    to_differentiate.aligned.str.get(
                        selected_additional_position))
            possible_additional_positions.remove(selected_additional_position)

    additional_positions = sorted(set(additional_positions))
    print(
        "Selected %d additional positions: " % len(additional_positions),
        additional_positions)

    extended_selected_positions = sorted(
        set(selected_positions).union(set(additional_positions)))
    print(
        "Extended selected positions (%d)" % len(extended_selected_positions),
        *extended_selected_positions)

    allele_sequences["sequence"] = allele_sequences.aligned.map(
        lambda s: "".join(s[p] for p in extended_selected_positions))

    allele_sequences[["sequence"]].to_csv(args.out_csv, index=True)
    print("Wrote: %s" % args.out_csv)


if __name__ == '__main__':
    run()
