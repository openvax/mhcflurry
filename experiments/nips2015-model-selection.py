#!/usr/bin/env python

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

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals
)
from os.path import join
import argparse
from time import time


import numpy as np
import pandas as pd

from mhcflurry.data_helpers import load_data
from mhcflurry.paths import (
    CLASS1_DATA_DIRECTORY
)

from model_configs import generate_all_model_configs
from model_selection import evaluate_model_config
from summarize_model_results import hyperparameter_performance

PETERS2009_CSV_FILENAME = "bdata.2009.mhci.public.1.txt"
PETERS2009_CSV_PATH = join(CLASS1_DATA_DIRECTORY, PETERS2009_CSV_FILENAME)

PETERS2013_CSV_FILENAME = "bdata.20130222.mhci.public.1.txt"
PETERS2013_CSV_PATH = join(CLASS1_DATA_DIRECTORY, PETERS2013_CSV_FILENAME)

COMBINED_CSV_FILENAME = "combined_human_class1_dataset.csv"
COMBINED_CSV_PATH = join(CLASS1_DATA_DIRECTORY, COMBINED_CSV_FILENAME)

parser = argparse.ArgumentParser()


def parse_int_list(string):
    substrings = [substring.strip() for substring in string.split(",")]
    return [int(substring) for substring in substrings if substring]


def parse_float_list(string):
    substrings = [substring.strip() for substring in string.split(",")]
    return [float(substring) for substring in substrings if substring]

parser.add_argument(
    "--binding-data-csv-path",
    default=PETERS2009_CSV_PATH,
    help="CSV file with 'mhc', 'peptide', 'peptide_length', 'meas' columns")

parser.add_argument(
    "--min-samples-per-allele",
    default=5,
    help="Don't train predictors for alleles with fewer samples than this",
    type=int)

parser.add_argument(
    "--results-filename",
    required=True,
    help="Write all hyperparameter/allele results to this filename")

parser.add_argument(
    "--cv-folds",
    default=5,
    type=int,
    help="Number cross-validation folds")

parser.add_argument(
    "--pretrain-epochs",
    default=[0, 10],
    type=parse_int_list,
    help="Number of pre-training epochs which use all allele data combined")

parser.add_argument(
    "--training-epochs",
    default=[200],
    type=parse_int_list,
    help="Number of passes over the dataset to perform during model fitting")

parser.add_argument(
    "--dropout",
    default=[0.0, 0.25],
    type=parse_float_list,
    help="Degree of dropout regularization to try in hyperparameter search")

parser.add_argument(
    "--minibatch-size",
    default=256,
    type=parse_int_list,
    help="How many samples to use in stochastic gradient estimation")

parser.add_argument(
    "--embedding-size",
    default=[0, 64],
    type=parse_int_list,
    help="Size of vector embedding dimension")

parser.add_argument(
    "--learning-rate",
    default=0.001,
    type=float,
    help="Learning rate for RMSprop")

parser.add_argument(
    "--hidden-layer-size",
    default=[50, 400],
    type=parse_int_list,
    help="Comma separated list of hidden layer sizes")

if __name__ == "__main__":
    args = parser.parse_args()
    configs = generate_all_model_configs(
        dropout_values=args.dropout,
        minibatch_sizes=args.minibatch_size,
        embedding_sizes=args.embedding_size,
        n_pretrain_epochs_values=args.pretrain_epochs,
        n_training_epochs_values=args.training_epochs,
        hidden_layer_sizes=args.hidden_layer_size)

    print("Total # configurations = %d" % len(configs))

    all_dataframes = []
    all_elapsed_times = []
    allele_datasets, _ = load_data(
        args.binding_data_csv_path,
        peptide_length=9,
        binary_encoding=False)
    for i, config in enumerate(configs):
        t_start = time()
        print("\n\n=== Config %d/%d: %s" % (i + 1, len(configs), config))
        result_df = evaluate_model_config(
            config,
            allele_datasets,
            min_samples_per_allele=args.min_samples_per_allele,
            cv_folds=args.cv_folds,
            learning_rate=args.learning_rate)
        n_rows = len(result_df)
        result_df["config_idx"] = [i] * n_rows
        for hyperparameter_name in config._fields:
            value = getattr(config, hyperparameter_name)
            result_df[hyperparameter_name] = [value] * n_rows
        # overwrite existing files for first config
        # only write column names for first batch of data
        # append results to CSV
        with open(args.results_filename, mode=("a" if i > 0 else "w")) as f:
            result_df.to_csv(f, index=False, header=(i == 0))
        all_dataframes.append(result_df)
        t_end = time()
        t_elapsed = t_end - t_start
        all_elapsed_times.append(t_elapsed)
        median_elapsed_time = np.median(all_elapsed_times)
        estimate_remaining = (len(configs) - i - 1) * median_elapsed_time
        print(
            "-- Time for config = %0.2fs, estimated remaining: %0.2f hours" % (
                t_elapsed,
                estimate_remaining / (60 * 60)))
    combined_df = pd.concat(all_dataframes)
    hyperparameter_performance(combined_df)
