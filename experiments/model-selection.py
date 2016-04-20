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
import argparse

from mhcflurry.data import load_allele_datasets


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
    evaluate_model_configs,
)

from summarize_model_results import hyperparameter_performance
from arg_parsing import parse_int_list, parse_float_list, parse_string_list
from dataset_paths import PETERS2009_CSV_PATH

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
    training_datasets = load_allele_datasets(
        args.binding_data_csv_path,
        max_ic50=args.max_ic50,
        peptide_length=9)
    combined_df = evaluate_model_configs(
        configs=configs,
        results_filename=args.output,
        train_fn=lambda config: evaluate_model_config_by_cross_validation(
            config,
            training_datasets,
            min_samples_per_allele=args.min_samples_per_allele,
            cv_folds=args.cv_folds))
    hyperparameter_performance(combined_df)
