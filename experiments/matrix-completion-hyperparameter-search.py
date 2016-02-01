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
For each allele, perform 5-fold cross validation to
    (1) complete the matrix of affinities to generate auxilliary training data
    (2) train multiple hyperparameter to find the best hyperparameters for
        each network
"""

import argparse
from collections import defaultdict

from fancyimpute import MICE, KNN, SimpleFill
import numpy as np
import pandas as pd
from mhcflurry.peptide_encoding import fixed_length_index_encoding
from mhcflurry.amino_acid import (
    common_amino_acids,
    amino_acids_with_unknown,
)
from mhcflurry import Class1BindingPredictor
from sklearn.cross_validation import StratifiedKFold

from dataset_paths import PETERS2009_CSV_PATH
from score_set import ScoreSet
from matrix_completion_helpers import load_data, evaluate_predictions

from arg_parsing import parse_int_list, parse_float_list

from matrix_completion_helpers import prune_data

parser = argparse.ArgumentParser()

parser.add_argument(
    "--binding-data-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--max-ic50",
    default=50000.0,
    type=float)

parser.add_argument(
    "--save-incomplete-affinity-matrix",
    default=None,
    help="Path to CSV which will contains the incomplete affinity matrix")

parser.add_argument(
    "--only-human",
    default=False,
    action="store_true")

parser.add_argument(
    "--output-file",
    default="hyperparameter-matrix-completion-results.csv")

parser.add_argument(
    "--n-folds",
    default=5,
    type=int,
    help="Number of cross-validation folds")

parser.add_argument(
    "--max-training-peptide-length",
    type=int,
    default=15)

parser.add_argument(
    "--verbose",
    default=False,
    action="store_true")

parser.add_argument(
    "--embedding-dim-sizes",
    default=[5, 10, 20],
    type=parse_int_list)

parser.add_argument(
    "--hidden-layer-sizes",
    default=[5, 20, 80],
    type=parse_int_list)

parser.add_argument(
    "--dropouts",
    default=[0.0, 0.25],
    type=parse_float_list)

parser.add_argument(
    "--activation-functions",
    default=["tanh"],
    type=lambda s: [si.strip() for si in s.split(",")])

parser.add_argument(
    "--training-epochs",
    type=int,
    default=100)

parser.add_argument(
    "--impute",
    default="mice",
    help="Use {'mice', 'knn', 'meanfill'} for imputing pre-training data")

parser.add_argument(
    "--unknown-amino-acids",
    default=False,
    action="store_true",
    help="When expanding 8mers into 9mers use 'X' instead of all possible AAs")


def print_length_distribution(peptides, values, name):
    print("Length distribution for %s (n=%d):" % (name, len(peptides)))
    grouped_affinity_dict = defaultdict(list)
    for p, v in zip(peptides, values):
        grouped_affinity_dict[len(p)].append(v)
    for length, affinities in sorted(grouped_affinity_dict.items()):
        print("%d => %d (mean affinity %0.4f)" % (
            length,
            len(affinities),
            np.mean(affinities)))


def print_full_dataset_length_distribution(
        pMHC_affinity_matrix,
        observed_mask,
        row_peptide_sequences):
    row_idx, col_idx = np.where(observed_mask)
    all_observed_peptides = []
    all_observed_values = []

    for i, j in zip(row_idx, col_idx):
        all_observed_peptides.append(row_peptide_sequences[i])
        all_observed_values.append(pMHC_affinity_matrix[i, j])

    print_length_distribution(all_observed_peptides, all_observed_values, "ALL")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    # initially load all the data, we're going to later prune it for matrix
    # completion
    pMHC_affinity_matrix, peptide_list, allele_list = load_data(
        binding_data_csv=args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human,
        min_observations_per_peptide=1,
        min_observations_per_allele=1)
    observed_mask = np.isfinite(pMHC_affinity_matrix)
    print_full_dataset_length_distribution(
        pMHC_affinity_matrix=pMHC_affinity_matrix,
        observed_mask=observed_mask,
        row_peptide_sequences=peptide_list)

    n_observed_per_allele = observed_mask.sum(axis=0)

    print("Loaded binding data, shape: %s, n_observed=%d/%d (%0.2f%%)" % (
        pMHC_affinity_matrix.shape,
        observed_mask.sum(),
        pMHC_affinity_matrix.size,
        100.0 * observed_mask.sum() / pMHC_affinity_matrix.size))
    if args.save_incomplete_affinity_matrix:
        print("Saving incomplete data to %s" % args.save_incomplete_affinity_matrix)
        df = pd.DataFrame(pMHC_affinity_matrix, columns=allele_list, index=peptide_list)
        df.to_csv(args.save_incomplete_affinity_matrix, index_label="peptide")

    scores = ScoreSet(
        index=[
            "dropout_probability",
            "embedding_dim_size",
            "hidden_layer_size",
            "activation"
        ])

    if args.unknown_amino_acids:
        index_encoding = amino_acids_with_unknown.index_encoding
    else:
        index_encoding = common_amino_acids.index_encoding

    impute_method_name = args.impute.lower()
    if impute_method_name.startswith("mice"):
        imputer = MICE(n_burn_in=5, n_imputations=20, min_value=0, max_value=1)
    elif impute_method_name.startswith("knn"):
        imputer = KNN(k=1, orientation="columns", print_interval=10)
    elif impute_method_name.startswith("mean"):
        imputer = SimpleFill("mean")
    else:
        raise ValueError("Invalid imputation method: %s" % impute_method_name)

    # to avoid recompiling and initializing a lot of neural networks
    # just save all the models up front along with their initial
    # weight matrices
    predictors = {}
    initial_weights = {}
    initial_optimizer_states = {}
    for dropout in args.dropouts:
        for embedding_dim_size in args.embedding_dim_sizes:
            for hidden_layer_size in args.hidden_layer_sizes:
                for activation in args.activation_functions:
                    key = "%f,%d,%d,%s" % (
                        dropout,
                        embedding_dim_size,
                        hidden_layer_size,
                        activation
                    )
                    if args.verbose:
                        print("-- Creating predictor for hyperparameters: %s" % key)
                    predictor = Class1BindingPredictor.from_hyperparameters(
                        embedding_output_dim=embedding_dim_size,
                        layer_sizes=[hidden_layer_size],
                        activation=activation,
                        output_activation="sigmoid",
                        dropout_probability=dropout,
                        verbose=args.verbose,
                        allow_unknown_amino_acids=args.unknown_amino_acids,
                        embedding_input_dim=21 if args.unknown_amino_acids else 20,
                    )
                    predictors[key] = predictor
                    initial_weights[key] = predictor.model.get_weights()
                    initial_optimizer_states[key] = predictor.model.optimizer.get_state()

    # want at least 5 samples in each fold of CV
    # to make meaningful estimates of accuracy
    min_samples_per_cv_fold = 5 * args.n_folds
    for allele_idx, allele in enumerate(allele_list):
        if n_observed_per_allele[allele_idx] < min_samples_per_cv_fold:
            print("-- Skipping allele %s which only has %d samples (need %d)" % (
                allele,
                n_observed_per_allele[allele_idx],
                min_samples_per_cv_fold))
            continue
        column = pMHC_affinity_matrix[:, allele_idx]
        observed_row_indices = np.where(observed_mask[:, allele_idx])[0]

        # drop indices which are for peptides that are extremely long,
        # like a 26-mer
        observed_row_indices = np.array([
            i
            for i in observed_row_indices
            if len(peptide_list[i]) <= args.max_training_peptide_length
        ])
        observed_values = column[observed_row_indices]
        std = np.std(observed_values)
        if std < 0.001:
            print("-- Skipping allele %s due to insufficient variance in affinities" % (
                allele))
            continue

        print(
            "Evaluating allele %s (n=%d)" % (
                allele,
                len(observed_row_indices)))
        median_value = np.median(observed_values)
        observed_peptides = [peptide_list[i] for i in observed_row_indices]
        print_length_distribution(observed_peptides, observed_values, name=allele)

        # k-fold cross validation stratified to keep an even balance of low
        # vs. high-binding peptides in each fold
        for fold_idx, (train_indices, test_indices) in enumerate(
                StratifiedKFold(
                    y=observed_values < median_value,
                    n_folds=args.n_folds,
                    shuffle=True)):
            train_peptides = [observed_peptides[i] for i in train_indices]
            train_values = [observed_values[i] for i in train_indices]
            train_dict = {k: v for (k, v) in zip(train_peptides, train_values)}

            test_peptides = [observed_peptides[i] for i in test_indices]
            test_values = [observed_values[i] for i in test_indices]
            test_dict = {k: v for (k, v) in zip(test_peptides, test_values)}

            # drop the test peptides from the full matrix and then
            # run completion on it to get synthesized affinities
            pMHC_affinity_matrix_fold = pMHC_affinity_matrix.copy()
            test_indices_among_all_rows = observed_row_indices[test_indices]
            pMHC_affinity_matrix_fold[test_indices_among_all_rows, allele_idx] = np.nan

            # drop peptides with fewer than 2 measurements and alleles
            # with fewer than 10 peptides
            pMHC_affinity_matrix_fold, all_peptides_fold, all_alleles_fold = prune_data(
                X=pMHC_affinity_matrix_fold,
                peptide_list=peptide_list,
                allele_list=allele_list,
                min_observations_per_peptide=2,
                min_observations_per_allele=min(10, min_samples_per_cv_fold))

            pMHC_affinity_matrix_fold_completed = imputer.complete(pMHC_affinity_matrix_fold)
            # keep synthetic data for 9mer peptides,
            # otherwise we can an explosion of low weight samples
            # that are expanded from e.g. 11mers
            # In the future: we'll use an neural network that
            # takes multiple input lengths
            ninemer_mask = np.array([len(p) == 9 for p in all_peptides_fold])
            ninemer_indices = np.where(ninemer_mask)[0]
            pMHC_affinity_matrix_fold_completed_9mers = \
                pMHC_affinity_matrix_fold_completed[ninemer_indices]
            if args.verbose:
                print("-- pMHC matrix for all peptides shape = %s" % (
                    pMHC_affinity_matrix_fold_completed.shape,))
                print("-- pMHC matrix for only 9mers shape = %s" % (
                    pMHC_affinity_matrix_fold_completed_9mers.shape,))
                print("-- 9mer indices n=%d max=%d" % (
                    len(ninemer_indices), ninemer_indices.max()))

            pretrain_peptides = [
                all_peptides_fold[i] for i in ninemer_indices
            ]
            X_pretrain = index_encoding(pretrain_peptides, 9)
            Y_pretrain = pMHC_affinity_matrix_fold_completed_9mers[
                :, allele_idx]

            X_train, training_row_peptides, training_counts = \
                fixed_length_index_encoding(
                    peptides=train_peptides,
                    desired_length=9,
                    allow_unknown_amino_acids=args.unknown_amino_acids)
            most_common_peptide_idx = np.argmax(training_counts)
            if args.verbose:
                print("-- Most common peptide in training data: %s (length=%d, count=%d)" % (
                    training_row_peptides[most_common_peptide_idx],
                    len(training_row_peptides[most_common_peptide_idx]),
                    training_counts[most_common_peptide_idx]))
            training_sample_weights = 1.0 / np.array(training_counts)
            Y_train = np.array([
                train_dict[p] for p in training_row_peptides])
            for key, predictor in predictors.items():

                print("\n-----")
                print(
                    ("Training model for %s (# peptides = %d, # samples=%d)"
                     " with parameters: %s") % (
                        allele,
                        len(train_peptides),
                        len(X_train),
                        key))
                print("-----")
                predictor.model.set_weights(initial_weights[key])
                predictor.model.optimizer.set_state(initial_optimizer_states[key])
                predictor.fit(
                    X=X_train,
                    Y=Y_train,
                    sample_weights=training_sample_weights,
                    X_pretrain=X_pretrain,
                    Y_pretrain=Y_pretrain,
                    n_training_epochs=args.training_epochs,
                    verbose=args.verbose)
                y_pred = predictor.predict_peptides_log_ic50(test_peptides)
                if args.verbose:
                    print("-- mean(Y) = %f, mean(Y_pred) = %f" % (
                        Y_train.mean(),
                        y_pred.mean()))
                mae, tau, auc, f1_score = evaluate_predictions(
                    y_true=test_values,
                    y_pred=y_pred,
                    max_ic50=args.max_ic50)
                scores.add_many(
                    key,
                    mae=mae,
                    tau=tau,
                    f1_score=f1_score,
                    auc=auc)

    scores.to_csv(args.output_file)
