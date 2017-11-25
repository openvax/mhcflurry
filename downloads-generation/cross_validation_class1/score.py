"""
Scoring script for cross-validation.
"""
import argparse
import sys
import collections

import pandas
from mhcflurry.scoring import make_scores


parser = argparse.ArgumentParser(usage = __doc__)

parser.add_argument(
    "input", metavar="INPUT.csv", help="Input CSV", nargs="+")

parser.add_argument(
    "--out-scores",
    metavar="RESULT.csv")

parser.add_argument(
    "--out-combined",
    metavar="COMBINED.csv")

parser.add_argument(
    "--out-summary",
    metavar="RESULT.csv")

def run(argv):
    args = parser.parse_args(argv)

    df = None
    for (i, filename) in enumerate(args.input):
        input_df = pandas.read_csv(filename)
        assert not input_df.mhcflurry_prediction.isnull().any()

        cols_to_merge = []
        input_df["prediction_%d" % i] = input_df.mhcflurry_prediction
        cols_to_merge.append(input_df.columns[-1])
        if 'mhcflurry_model_single_0' in input_df.columns:
            input_df["prediction_single_%d" % i] = input_df.mhcflurry_model_single_0
            cols_to_merge.append(input_df.columns[-1])

        if df is None:
            df = input_df[
                ["allele", "peptide", "measurement_value"] + cols_to_merge
            ].copy()
        else:
            df = pandas.merge(
                df,
                input_df[['allele', 'peptide'] + cols_to_merge],
                on=['allele', 'peptide'],
                how='outer')

    print("Loaded data:")
    print(df.head(5))

    if args.out_combined:
        df.to_csv(args.out_combined, index=False)
        print("Wrote: %s" % args.out_combined)

    prediction_cols = [
        c
        for c in df.columns
        if c.startswith("prediction_")
    ]

    scores_rows = []
    for (allele, allele_df) in df.groupby("allele"):
        for prediction_col in prediction_cols:
            sub_df = allele_df.loc[~allele_df[prediction_col].isnull()]
            scores = collections.OrderedDict()
            scores['allele'] = allele
            scores['fold'] = prediction_col.replace("prediction_", "").replace("single_", "")
            scores['kind'] = "single" if "single" in prediction_col else "ensemble"
            scores['train_size'] = allele_df[prediction_col].isnull().sum()
            scores['test_size'] = len(sub_df)
            scores.update(
                make_scores(
                    sub_df.measurement_value, sub_df[prediction_col]))
            scores_rows.append(scores)
    scores_df = pandas.DataFrame(scores_rows)
    print(scores_df)

    if args.out_scores:
        scores_df.to_csv(args.out_scores, index=False)
        print("Wrote: %s" % args.out_scores)

    summary_df = scores_df.groupby(["allele", "kind"])[
        ["train_size", "test_size", "auc", "f1", "tau"]
    ].mean().reset_index()
    print("Summary:")
    print(summary_df)

    if args.out_summary:
        summary_df.to_csv(args.out_summary, index=False)
        print("Wrote: %s" % args.out_summary)

if __name__ == '__main__':
    run(sys.argv[1:])

