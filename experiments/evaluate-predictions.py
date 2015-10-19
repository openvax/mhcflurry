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

"""
Compute accuracy, AUC, and F1 score for allele-specific test datasets
"""

from os import listdir
from os.path import join
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from model_selection_helpers import f1_score

parser = ArgumentParser()

parser.add_argument(
    "--test-data-dir",
    help="Directory which contains one CSV file per allele",
    required=True)

parser.add_argument(
    "--true-ic50-column-name",
    default="meas")

parser.add_argument(
    "--peptide-sequence-column-name",
    default="sequence")

parser.add_argument(
    "--peptide-length-column-name",
    default="length")

if __name__ == "__main__":
    args = parser.parse_args()

    # mapping from predictor names to dictionaries
    results = defaultdict(lambda: OrderedDict([
        ("allele", []),
        ("length", []),
        ("auc", []),
        ("accuracy", []),
        ("f1", [])]
    ))

    for filename in listdir(args.test_data_dir):
        filepath = join(args.test_data_dir, filename)
        parts = filename.split(".")
        if len(parts) != 2:
            print("Skipping %s" % filepath)
            continue
        allele, ext = parts
        if ext != "csv":
            print("Skipping %s, only reading CSV files" % filepath)
            continue
        df = pd.read_csv(filepath)
        columns = set(df.columns)
        drop_columns = {
            args.true_ic50_column_name,
            args.peptide_length_column_name,
            args.peptide_sequence_column_name,
        }
        predictor_names = columns.difference(drop_columns)
        true_ic50 = df[args.true_ic50_column_name]
        true_label = true_ic50 <= 500
        n = len(df)
        print("%s (total = %d, n_pos = %d, n_neg = %d)" % (
            allele,
            n,
            true_label.sum(),
            n - true_label.sum()))

        for predictor in sorted(predictor_names):
            pred_ic50 = df[predictor]
            pred_label = pred_ic50 <= 500
            if true_label.std() == 0:
                # can't compute AUC from single class
                auc = np.nan
            else:
                # using negative IC50 since it's inversely related to binding
                auc = roc_auc_score(true_label, -pred_ic50)

            f1 = f1_score(true_label, pred_label)
            accuracy = np.mean(true_label == pred_label)
            print("-- %s AUC=%0.4f, acc=%0.4f, F1=%0.4f" % (
                predictor,
                auc,
                accuracy,
                f1))
            results[predictor]["allele"].append(allele)
            results[predictor]["length"].append(n)
            results[predictor]["f1"].append(f1)
            results[predictor]["accuracy"].append(accuracy)
            results[predictor]["auc"].append(auc)

    print("\n === Aggregate Results ===\n")
    for (predictor, data) in sorted(results.items()):
        df = pd.DataFrame(data)
        print(predictor)
        aucs = df["auc"]
        auc_lower = aucs.quantile(0.25)
        auc_upper = aucs.quantile(0.75)
        auc_iqr = auc_upper - auc_lower
        print("-- AUC: median=%0.4f, mean=%0.4f, iqr=%0.4f" % (
            aucs.median(),
            aucs.mean(),
            auc_iqr))
        f1s = df["f1"]
        f1_lower = f1s.quantile(0.25)
        f1_upper = f1s.quantile(0.75)
        f1_iqr = f1_upper - f1_lower
        print("-- F1: median=%0.4f, mean=%0.4f, iqr=%0.4f" % (
            f1s.median(),
            f1s.mean(),
            f1_iqr))
