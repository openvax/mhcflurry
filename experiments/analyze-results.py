#!/usr/bin/env python
#
# Copyright (c) 2015. Mount Sinai School of Medicine
#
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

from argparse import ArgumentParser

import pandas as pd
import numpy as np

from summarize_model_results import (
    hyperparameter_performance,
    hyperparameter_score_difference_hypothesis_tests,
)

parser = ArgumentParser()

parser.add_argument(
    "filename",
    help="CSV with results from hyperparameter search")

parser.add_argument(
    "--latex-table-filename",
    default=None,
    help="Path to file where we should write a LaTeX table summarizing results")


def infer_dtypes(df):
    column_names = list(df.columns)
    for column_name in column_names:
        column_values = np.array(df[column_name])
        if "object" in str(column_values.dtype):
            if any("." in value for value in column_values):
                print("Converting %s to float" % column_name)
                df[column_name] = column_values.astype(float)
            elif all(value.isdigit() for value in column_values):
                print("Converting %s to int" % column_name)
                df[column_name] = column_values.astype(int)
    return df


def generate_latex_table(results_dict, caption="CAPTION", label="LABEL"):
    template = """
        \\begin{table}[htbp]
            \\label{%(label)s}
            \\begin{center}
            \\begin{tabular}{cccc}
            \\multicolumn{1}{c}{\\bf Hyperparameter}  &
                \\multicolumn{1}{c}{\\bf Better} &
                \\multicolumn{1}{c}{\\bf Worse} &
                \\multicolumn{1}{c}{\\bf p-value}\\\\
            \hline \\\\
            %(content)s
            \\end{tabular}
            \\caption{%(caption)s}
            \\end{center}
        \\end{table}
    """
    lines = []
    for hyperparameter_name, pairwise_results in sorted(
            results_dict.items(),
            key=lambda x: min(result.p for result in x[1])):

        for result in sorted(pairwise_results, key=lambda result: result.p):
            log_p = np.log10(result.p)
            if log_p < -4:
                p_str = "10^{%d}" % int(log_p)
            else:
                p_str = "%0.4f" % result.p
            lines.append(
                "\t\t%s &  {\\bf %s} & %s & $%s$" % (
                    hyperparameter_name.replace("_", " "),
                    str(result.better_value).replace("_", " "),
                    str(result.worse_value).replace("_", " "),
                    p_str))
    #
    # Input Encoding              & 1-of-k binary, vector embedding \\
    #    \# Pre-training Epochs      & 0, 10 \\
    #    Hidden Layer Size & 50, 400 \\

    return template % {
        "label": label,
        "caption": caption,
        "content": "\\\\\n".join(lines)
    }


if __name__ == "__main__":
    args = parser.parse_args()
    results = pd.read_csv(args.filename, sep=",", header=0)
    results = infer_dtypes(results)
    hyperparameter_columns = list(sorted({
        name for name in results.columns
        if "idx" not in name
        and "_mean" not in name
        and "_median" not in name
        and "_min" not in name
        and "_max" not in name
        and "_iqr" not in name
        and "_std" not in name
        and "_amin" not in name
        and "_amax" not in name
        and "dataset_size" not in name
        and "allele_name" not in name
    }))
    max_score = 0
    best_f1 = 0
    best_auc = 0
    best_config = None
    for _, group in results.groupby("config_idx"):
        auc = group["auc_mean"]
        f1 = group["f1_mean"]
        auc_std = group["auc_std"]
        f1_std = group["f1_std"]
        combined_scores = np.sqrt(np.array(auc * f1))
        score = np.mean(combined_scores)
        score_std = np.std(combined_scores)
        config_row = group[hyperparameter_columns].iloc[0]
        print(config_row)
        print("--> AUC: %0.4f +/- %0.4f" % (
            auc.mean(),
            auc.std()))
        print("--> F1: %0.4f +/- %0.4f" % (
            f1.mean(),
            f1.std()))

        print("--> combined: %0.4f +/- %0.4f" % (
            score,
            score_std))

        if score > max_score:
            best_auc = auc.mean()
            best_f1 = f1.mean()
            max_score = score
            best_config = config_row
    print("Best score: %0.4f, AUC=%0.4f, f1=%0.4f, hyperparameters:\n%s" % (
        max_score, best_auc, best_f1, best_config))
    print("=== Score Distributions For Hyperparameters ===")
    hyperparameter_performance(results)
    print("\n\n=== Hyperparameter Value Comparisons ===")
    results = hyperparameter_score_difference_hypothesis_tests(results)
    print(generate_latex_table(results))
