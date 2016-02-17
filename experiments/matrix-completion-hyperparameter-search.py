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

from fancyimpute import MICE, KNN, SimpleFill, IterativeSVD, SoftImpute
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

from matrix_completion_helpers import load_data, evaluate_predictions

from arg_parsing import parse_int_list, parse_float_list, parse_string_list

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
    default=[10, 20, 40],
    type=parse_int_list)

parser.add_argument(
    "--first-hidden-layer-sizes",
    default=[25, 50, 100],
    type=parse_int_list)


parser.add_argument(
    "--second-hidden-layer-sizes",
    default=[0],
    type=parse_int_list)


parser.add_argument(
    "--dropouts",
    default=[0.0],
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
    help="Use {'mice', 'knn', 'meanfill', 'none', 'svt', 'svd'} for imputing pre-training data")

parser.add_argument("--batch-size", type=int, default=64)

parser.add_argument(
    "--unknown-amino-acids",
    default=False,
    action="store_true",
    help="When expanding 8mers into 9mers use 'X' instead of all possible AAs")

parser.add_argument(
    "--pretrain-only-9mers",
    default=False,
    action="store_true",
    help="Exclude expanded samples from e.g. 8mers or 10mers")

parser.add_argument(
    "--alleles",
    default=None,
    type=parse_string_list,
    help="Only evaluate these alleles (by default use all in the dataset)")

parser.add_argument("--min-samples-per-cv-fold", type=int, default=5)


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

    if args.verbose:
        print("\n----\n# observed per allele:")
        for allele_idx, allele in sorted(enumerate(allele_list), key=lambda x: x[1]):
            print("--- %s: %d" % (allele, n_observed_per_allele[allele_idx]))

    if args.save_incomplete_affinity_matrix:
        print("Saving incomplete data to %s" % args.save_incomplete_affinity_matrix)
        df = pd.DataFrame(pMHC_affinity_matrix, columns=allele_list, index=peptide_list)
        df.to_csv(args.save_incomplete_affinity_matrix, index_label="peptide")

    if args.output_file:
        output_file = open(args.output_file, "w")
        fields = [
            "allele",
            "cv_fold",
            "peptide_count",
            "sample_count",
            "dropout_probability",
            "embedding_dim_size",
            "hidden_layer_size1",
            "hidden_layer_size2",
            "activation"
            "mae",
            "tau",
            "auc",
            "f1"
        ]
        header_line = ",".join(fields)
        output_file.write(header_line + "\n")
    else:
        output_file = None
    if args.unknown_amino_acids:
        index_encoding = amino_acids_with_unknown.index_encoding
    else:
        index_encoding = common_amino_acids.index_encoding

    impute_method_name = args.impute.lower()
    if impute_method_name.startswith("mice"):
        imputer = MICE(n_burn_in=5, n_imputations=20, min_value=0, max_value=1)
    elif impute_method_name.startswith("knn"):
        imputer = KNN(k=1, orientation="columns", print_interval=10)
    elif impute_method_name.startswith("svd"):
        imputer = IterativeSVD(rank=10)
    elif impute_method_name.startswith("svt"):
        imputer = SoftImpute()
    elif impute_method_name.startswith("mean"):
        imputer = SimpleFill("mean")
    elif impute_method_name == "none":
        imputer = None
    else:
        raise ValueError("Invalid imputation method: %s" % impute_method_name)

    predictors = {}
    initial_weights = {}
    initial_optimizer_states = {}

    def generate_predictors():
        """
        Generator of all possible predictors generated by combinations of
        hyperparameters.

        To avoid recompiling and initializing a lot of neural networks
        just save all the models up front along with their initial weight matrices
        """
        for dropout in args.dropouts:
            for embedding_dim_size in args.embedding_dim_sizes:
                for hidden_layer_size1 in args.first_hidden_layer_sizes:
                    for hidden_layer_size2 in args.second_hidden_layer_sizes:
                        for activation in args.activation_functions:
                            key = "%0.2f,%d,%d,%d,%s" % (
                                dropout,
                                embedding_dim_size,
                                hidden_layer_size1,
                                hidden_layer_size2,
                                activation
                            )
                            if key not in predictors:
                                layer_sizes = [hidden_layer_size1]
                                if hidden_layer_size2:
                                    layer_sizes.append(hidden_layer_size2)

                                predictor = Class1BindingPredictor.from_hyperparameters(
                                    embedding_output_dim=embedding_dim_size,
                                    layer_sizes=layer_sizes,
                                    activation=activation,
                                    output_activation="sigmoid",
                                    dropout_probability=dropout,
                                    verbose=args.verbose,
                                    allow_unknown_amino_acids=args.unknown_amino_acids,
                                    embedding_input_dim=21 if args.unknown_amino_acids else 20,
                                )
                                weights = predictor.model.get_weights()
                                opt_state = predictor.model.optimizer.get_state()
                                predictors[key] = predictor
                                initial_weights[key] = weights
                                initial_optimizer_states[key] = opt_state
                            else:
                                predictor = predictors[key]
                                weights = initial_weights[key]
                                opt_state = initial_optimizer_states[key]
                                # reset the predictor to its initial condition
                                predictor.model.set_weights([
                                    w.copy() for w in weights])
                                predictor.model.optimizer.set_state([
                                    s.copy() for s in opt_state])
                            yield (key, predictor)

    min_samples_per_allele = args.min_samples_per_cv_fold * args.n_folds
    for allele_idx, allele in enumerate(allele_list):
        if args.alleles and not any(
                pattern in allele for pattern in args.alleles):
            # if user specifies an allele list then skip anything which isn't included
            continue
        if n_observed_per_allele[allele_idx] < min_samples_per_allele:
            print("-- Skipping allele %s which only has %d samples (need %d)" % (
                allele,
                n_observed_per_allele[allele_idx],
                min_samples_per_allele))
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
        ic50_values = args.max_ic50 ** (1 - observed_values)

        below_500nM = ic50_values <= 500
        if below_500nM.all():
            print("-- Skipping %s due to all negative predictions" % allele)
            continue
        if below_500nM.sum() < args.n_folds:
            print("-- Skipping %s due to insufficient positive examples (%d)" % (
                allele,
                below_500nM.sum()))
            continue

        observed_peptides = [peptide_list[i] for i in observed_row_indices]
        print_length_distribution(observed_peptides, observed_values, name=allele)

        # k-fold cross validation stratified to keep an even balance of low
        # vs. high-binding peptides in each fold
        for fold_idx, (train_indices, test_indices) in enumerate(StratifiedKFold(
                y=below_500nM,
                n_folds=args.n_folds,
                shuffle=True)):

            train_peptides_fold = [observed_peptides[i] for i in train_indices]
            train_values_fold = [observed_values[i] for i in train_indices]
            train_dict_fold = {k: v for (k, v) in zip(train_peptides_fold, train_values_fold)}

            test_peptides = [observed_peptides[i] for i in test_indices]
            test_values = [observed_values[i] for i in test_indices]
            test_dict = {k: v for (k, v) in zip(test_peptides, test_values)}
            if imputer is None:
                X_pretrain = np.array([], dtype=int).reshape((0, 9))
                Y_pretrain = np.array([], dtype=float)
                pretrain_sample_weights = np.array([], dtype=float)
            else:
                # drop the test peptides from the full matrix and then
                # run completion on it to get synthesized affinities
                pMHC_affinities_fold_incomplete = pMHC_affinity_matrix.copy()
                test_indices_among_all_rows = observed_row_indices[test_indices]
                pMHC_affinities_fold_incomplete[
                    test_indices_among_all_rows, allele_idx] = np.nan

                # drop peptides with fewer than 2 measurements and alleles
                # with fewer than 10 peptides
                pMHC_affinities_fold_pruned, pruned_peptides_fold, pruned_alleles_fold = \
                    prune_data(
                        X=pMHC_affinities_fold_incomplete,
                        peptide_list=peptide_list,
                        allele_list=allele_list,
                        min_observations_per_peptide=2,
                        min_observations_per_allele=args.min_samples_per_cv_fold)

                pMHC_affinities_fold_pruned_imputed = imputer.complete(
                    pMHC_affinities_fold_pruned)

                if args.pretrain_only_9mers:
                    # if we're expanding 8mers to >100 9mers
                    # then the pretraining dataset becomes
                    # unmanageably large, so let's only use 9mers for pre-training

                    # In the future: we'll use an neural network that
                    # takes multiple input lengths
                    ninemer_mask = np.array([len(p) == 9 for p in pruned_peptides_fold])
                    ninemer_indices = np.where(ninemer_mask)[0]
                    pMHC_affinities_fold_pruned_imputed = \
                        pMHC_affinities_fold_pruned_imputed[ninemer_indices]
                    pruned_peptides_fold = [pruned_peptides_fold[i] for i in ninemer_indices]

                # since we may have dropped some of the columns in the completed
                # pMHC matrix need to find which column corresponds to the
                # same name we're currently predicting
                if allele in pruned_alleles_fold:
                    pMHC_fold_allele_idx = pruned_alleles_fold.index(allele)
                    pMHC_allele_values = pMHC_affinities_fold_pruned_imputed[
                        :, pMHC_fold_allele_idx]
                    if pMHC_allele_values.std() == 0:
                        print("WARNING: unexpected zero-variance in pretraining affinity values")
                else:
                    print(
                        "WARNING: Couldn't find allele %s in pre-training matrix" % allele)
                    column = pMHC_affinities_fold_incomplete[:, allele_idx]
                    column_mean = np.nanmean(column)
                    print("-- Setting pre-training target value to nanmean = %f" % column_mean)
                    pMHC_allele_values = np.ones(len(pruned_peptides_fold)) * column_mean

                assert len(pruned_peptides_fold) == len(pMHC_allele_values)

                # dictionary mapping peptides to imputed affinity values
                pretrain_dict = {
                    pi: yi
                    for (pi, yi)
                    in zip(pruned_peptides_fold, pMHC_allele_values)
                }

                X_pretrain, pretrain_row_peptides, pretrain_counts = \
                    fixed_length_index_encoding(
                        peptides=pruned_peptides_fold,
                        desired_length=9,
                        allow_unknown_amino_acids=args.unknown_amino_acids)
                pretrain_sample_weights = 1.0 / np.array(pretrain_counts)
                Y_pretrain = np.array(
                    [pretrain_dict[p] for p in pretrain_row_peptides])

            X_train, training_row_peptides, training_counts = \
                fixed_length_index_encoding(
                    peptides=train_peptides_fold,
                    desired_length=9,
                    allow_unknown_amino_acids=args.unknown_amino_acids)
            training_sample_weights = 1.0 / np.array(training_counts)
            Y_train = np.array([train_dict_fold[p] for p in training_row_peptides])

            n_pretrain = len(X_pretrain)
            n_train_unique = len(train_peptides_fold)
            n_train = len(X_train)
            for key, predictor in generate_predictors():
                print("\n-----")
                print(
                    ("Training CV fold %d/%d for %s "
                     "(# peptides = %d, # samples=%d, # pretrain samples=%d)"
                     " with parameters: %s") % (
                        fold_idx + 1,
                        args.n_folds,
                        allele,
                        n_train_unique,
                        n_train,
                        n_pretrain,
                        key))
                print("-----")
                predictor.fit(
                    X=X_train,
                    Y=Y_train,
                    sample_weights=training_sample_weights,
                    X_pretrain=X_pretrain,
                    Y_pretrain=Y_pretrain,
                    pretrain_sample_weights=pretrain_sample_weights,
                    n_training_epochs=args.training_epochs,
                    verbose=args.verbose,
                    batch_size=args.batch_size)
                y_pred = predictor.predict_peptides_log_ic50(test_peptides)
                if args.verbose:
                    print("-- mean(Y) = %f, mean(Y_pred) = %f" % (
                        Y_train.mean(),
                        y_pred.mean()))
                mae, tau, auc, f1_score = evaluate_predictions(
                    y_true=test_values,
                    y_pred=y_pred,
                    max_ic50=args.max_ic50)

                cv_fold_field_values = [
                    allele,
                    str(fold_idx),
                    str(n_train_unique),
                    str(n_train),
                ]
                accuracy_field_values = [
                    "%0.4f" % mae,
                    "%0.4f" % tau,
                    "%0.4f" % auc,
                    "%0.4f" % f1_score
                ]
                output_line = (
                    ",".join(cv_fold_field_values) +
                    "," + key +
                    "," + ",".join(accuracy_field_values) +
                    "\n"
                )
                print("CV fold result: %s" % output_line)
                if output_file:
                    output_file.write(output_line + "\n")
                    output_file.flush()
