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

from model_configs import (
    generate_all_model_configs,
    HIDDEN_LAYER_SIZES,
    INITILIZATION_METHODS,
    ACTIVATIONS,
    MAX_IC50_VALUES,
    EMBEDDING_SIZES,
    N_TRAINING_EPOCHS,
    N_PRETRAIN_EPOCHS,
    MINIBATCH_SIZES,
    DROPOUT_VALUES,
    LEARNING_RATES,
    OPTIMIZERS
)
from model_selection_helpers import (
    evaluate_model_config_by_cross_validation,
)

from summarize_model_results import hyperparameter_performance
from arg_parsing import parse_int_list, parse_float_list, parse_string_list


PETERS2009_CSV_FILENAME = "bdata.2009.mhci.public.1.txt"
PETERS2009_CSV_PATH = join(CLASS1_DATA_DIRECTORY, PETERS2009_CSV_FILENAME)

PETERS2013_CSV_FILENAME = "bdata.20130222.mhci.public.1.txt"
PETERS2013_CSV_PATH = join(CLASS1_DATA_DIRECTORY, PETERS2013_CSV_FILENAME)

BLIND_2013_CSV_FILENAME = "bdata.2013.mhci.public.blind.1.txt"
BLIND_2013_CSV_PATH = join(CLASS1_DATA_DIRECTORY, BLIND_2013_CSV_FILENAME)

COMBINED_CSV_FILENAME = "combined_human_class1_dataset.csv"
COMBINED_CSV_PATH = join(CLASS1_DATA_DIRECTORY, COMBINED_CSV_FILENAME)

parser = argparse.ArgumentParser()


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
    "--output",
    required=True,
    help="Write all hyperparameter/allele results to this filename")

parser.add_argument(
    "--cv-folds",
    default=5,
    type=int,
    help="Number cross-validation folds")

parser.add_argument(
    "--pretrain-epochs",
    default=N_PRETRAIN_EPOCHS,
    type=parse_int_list,
    help="Number of pre-training epochs which use all allele data combined")

parser.add_argument(
    "--training-epochs",
    default=N_TRAINING_EPOCHS,
    type=parse_int_list,
    help="Number of passes over the dataset to perform during model fitting")

parser.add_argument(
    "--dropout",
    default=DROPOUT_VALUES,
    type=parse_float_list,
    help="Degree of dropout regularization to try in hyperparameter search")

parser.add_argument(
    "--minibatch-size",
    default=MINIBATCH_SIZES,
    type=parse_int_list,
    help="How many samples to use in stochastic gradient estimation")

parser.add_argument(
    "--embedding-size",
    default=EMBEDDING_SIZES,
    type=parse_int_list,
    help="Size of vector embedding dimension")

parser.add_argument(
    "--learning-rate",
    default=LEARNING_RATES,
    type=parse_float_list,
    help="Learning rate for RMSprop")

parser.add_argument(
    "--hidden-layer-size",
    default=HIDDEN_LAYER_SIZES,
    type=parse_int_list,
    help="Comma separated list of hidden layer sizes")


parser.add_argument(
    "--max-ic50",
    default=MAX_IC50_VALUES,
    type=parse_float_list,
    help="Comma separated list of maximum predicted IC50 values")


parser.add_argument(
    "--init",
    default=INITILIZATION_METHODS,
    type=parse_string_list,
    help="Comma separated list of initialization methods")

parser.add_argument(
    "--activation",
    default=ACTIVATIONS,
    type=parse_string_list,
    help="Comma separated list of activation functions")

parser.add_argument(
    "--optimizer",
    default=OPTIMIZERS,
    type=parse_string_list,
    help="Comma separated list of optimization methods")


def evaluate_model_configs(configs, results_filename, train_fn):
    all_dataframes = []
    all_elapsed_times = []
    for i, config in enumerate(configs):
        t_start = time()
        print("\n\n=== Config %d/%d: %s" % (i + 1, len(configs), config))
        result_df = train_fn(config)
        n_rows = len(result_df)
        result_df["config_idx"] = [i] * n_rows
        for hyperparameter_name in config._fields:
            value = getattr(config, hyperparameter_name)
            result_df[hyperparameter_name] = [value] * n_rows
        # overwrite existing files for first config
        # only write column names for first batch of data
        # append results to CSV
        with open(results_filename, mode=("a" if i > 0 else "w")) as f:
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
    return pd.concat(all_dataframes)


if __name__ == "__main__":
    args = parser.parse_args()
    configs = generate_all_model_configs(
        activations=args.activation,
        init_methods=args.init,
        max_ic50_values=args.max_ic50,
        dropout_values=args.dropout,
        minibatch_sizes=args.minibatch_size,
        embedding_sizes=args.embedding_size,
        n_pretrain_epochs_values=args.pretrain_epochs,
        n_training_epochs_values=args.training_epochs,
        hidden_layer_sizes=args.hidden_layer_size,
        learning_rates=args.learning_rate,
        optimizers=args.optimizer)

    print("Total # configurations = %d" % len(configs))
    training_datasets, _ = load_data(
        args.binding_data_csv_path,
        peptide_length=9,
        binary_encoding=False)
    combined_df = evaluate_model_configs(
        configs=configs,
        results_filename=args.output,
        train_fn=lambda config: evaluate_model_config_by_cross_validation(
            config,
            training_datasets,
            min_samples_per_allele=args.min_samples_per_allele,
            cv_folds=args.cv_folds))
    hyperparameter_performance(combined_df)
