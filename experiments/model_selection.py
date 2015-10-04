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
from collections import OrderedDict
import logging

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.cross_validation
from sklearn.cross_validation import KFold

from mhcflurry.common import normalize_allele_name
from mhcflurry.feedforward import make_embedding_network, make_hotshot_network
from mhcflurry.data_helpers import indices_to_hotshot_encoding


def score_predictions(predicted_log_ic50, true_label, max_ic50):
    """Computes accuracy, AUC, and F1 score of predictions"""
    auc = sklearn.metrics.roc_auc_score(true_label, predicted_log_ic50)
    ic50_pred = max_ic50 ** (1.0 - predicted_log_ic50)
    label_pred = (ic50_pred <= 500)
    same_mask = true_label == label_pred
    accuracy = np.mean(same_mask)
    tp = (true_label & label_pred).sum()
    fp = ((~true_label) & label_pred).sum()
    tn = ((~true_label) & (~label_pred)).sum()
    fn = (true_label & (~label_pred)).sum()
    sensitivity = (tp / float(tp + fn)) if (tp + fn) > 0 else 0.0
    precision = (tp / float(tp + fp)) if (tp + fp) > 0 else 0.0
    if (precision + sensitivity) > 0:
        f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0.0
    # sanity check that we're computing accuracy correctly
    accuracy_estimate2 = (tp + tn) / float(tp + fp + tn + fn)
    if abs(accuracy - accuracy_estimate2) > 0.00001:
        logging.warn(
            "!!! Conflicting accuracy estimates! (%0.5f vs. %0.5f)" % (
                accuracy, accuracy_estimate2))
    return accuracy, auc, f1_score


def kfold_cross_validation_for_single_allele(
        allele_name,
        model,
        X,
        Y,
        ic50,
        n_training_epochs=100,
        cv_folds=5,
        max_ic50=5000,
        minibatch_size=512):
    """
    Estimate the per-allele AUC score of a model via k-fold cross-validation.
    Returns the per-fold AUC scores and accuracies.
    """
    n_samples = len(Y)
    initial_weights = [w.copy() for w in model.get_weights()]
    fold_aucs = []
    fold_accuracies = []
    fold_f1_scores = []

    for cv_iter, (train_idx, test_idx) in enumerate(KFold(
            n=n_samples,
            n_folds=cv_folds,
            shuffle=True,
            random_state=0)):
        X_train, Y_train = X[train_idx, :], Y[train_idx]
        X_test = X[test_idx, :]
        ic50_test = ic50[test_idx]
        label_test = ic50_test <= 500
        if label_test.all() or not label_test.any():
            print(
                "Skipping CV iter %d of %s since all outputs are the same" % (
                    cv_iter, allele_name))
            continue
        model.set_weights(initial_weights)

        model.fit(
            X_train,
            Y_train,
            nb_epoch=n_training_epochs,
            verbose=0,
            batch_size=minibatch_size)

        pred = model.predict(X_test).flatten()
        accuracy, auc, f1_score = score_predictions(pred, label_test, max_ic50)
        print(
            "-- %d/%d: AUC: %0.5f" % (
                cv_iter + 1,
                cv_folds,
                auc))
        print(
            "-- %d/%d Accuracy: %0.5f (baseline %0.5f)" % (
                cv_iter + 1,
                cv_folds,
                accuracy,
                max(label_test.mean(), 1.0 - label_test.mean())))
        print(
            "-- %d/%d F1-score: %0.5f" % (
                cv_iter + 1,
                cv_folds,
                f1_score))
        fold_aucs.append(auc)
        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1_score)
    return fold_aucs, fold_accuracies, fold_f1_scores


def leave_out_allele_cross_validation(
        model,
        allele_datasets,
        max_ic50,
        binary_encoding=False,
        n_pretrain_epochs=0,
        n_training_epochs=100,
        min_samples_per_allele=5,
        cv_folds=5,
        minibatch_size=128):
    """
    Fit the model for every allele in the dataset and return a DataFrame
    with the following columns:
            allele_name
            dataset_size
            auc_mean
            auc_median
            auc_std
            auc_min
            auc_max
            accuracy_mean
            accuracy_median
            accuracy_std
            accuracy_min
            accuracy_max
            f1_mean
            f1_median
            f1_std
            f1_min
            f1_max
    """
    result_dict = OrderedDict([
        ("allele_name", []),
        ("dataset_size", []),
    ])
    for score_name in ["auc", "accuracy", "f1"]:
        for statistic in ["mean", "median", "std", "min", "max"]:
            result_dict["%s_%s" % (score_name, statistic)] = []

    initial_weights = [w.copy() for w in model.get_weights()]
    for allele_name, dataset in sorted(
            allele_datasets.items(), key=lambda pair: pair[0]):
        # Want alleles to be 4-digit + gene name e.g. C0401
        if allele_name.isdigit() or len(allele_name) < 5:
            print("Skipping allele %s" % (allele_name,))
            continue
        allele_name = normalize_allele_name(allele_name)
        X_allele = dataset.X
        n_samples_allele = X_allele.shape[0]
        if n_samples_allele < min_samples_per_allele:
            print("Skipping allele %s due to too few samples: %d" % (
                allele_name, n_samples_allele))
            continue
        if binary_encoding:
            X_allele = indices_to_hotshot_encoding(X_allele, n_indices=20)

        ic50_allele = dataset.ic50
        Y_allele = 1.0 - np.minimum(1.0, np.log(ic50_allele) / np.log(max_ic50))
        model.set_weights(initial_weights)
        if n_pretrain_epochs > 0:
            X_other_alleles = np.vstack([
                other_dataset.X
                for (other_allele, other_dataset) in allele_datasets.items()
                if normalize_allele_name(other_allele) != allele_name])
            if binary_encoding:
                X_other_alleles = indices_to_hotshot_encoding(
                    X_other_alleles, n_indices=20)
            ic50_other_alleles = np.concatenate([
                other_dataset.ic50 for (other_allele, other_dataset)
                in allele_datasets.items()
                if normalize_allele_name(other_allele) != allele_name])
            Y_other_alleles = 1.0 - np.minimum(
                1.0,
                np.log(ic50_other_alleles) / np.log(max_ic50))
            print("Pre-training X shape: %s" % (X_other_alleles.shape,))
            print("Pre-training Y shape: %s" % (Y_other_alleles.shape,))
            model.fit(
                X_other_alleles,
                Y_other_alleles,
                nb_epoch=n_pretrain_epochs,
                batch_size=minibatch_size,
                verbose=0)
        print("Cross-validation for %s (%d):" % (allele_name, len(Y_allele)))
        aucs, accuracies, f1_scores = kfold_cross_validation_for_single_allele(
            allele_name=allele_name,
            model=model,
            X=X_allele,
            Y=Y_allele,
            ic50=ic50_allele,
            n_training_epochs=n_training_epochs,
            cv_folds=cv_folds,
            max_ic50=max_ic50,
            minibatch_size=minibatch_size)
        if len(aucs) == 0:
            print("Skipping allele %s" % allele_name)
            continue
        result_dict["allele_name"].append(allele_name)
        result_dict["dataset_size"].append(len(ic50_allele))
        for (name, values) in [
                ("auc", aucs),
                ("accuracy", accuracies),
                ("f1", f1_scores)]:
            result_dict["%s_mean" % name].append(np.mean(values))
            result_dict["%s_median" % name].append(np.median(values))
            result_dict["%s_std" % name].append(np.std(values))
            result_dict["%s_min" % name].append(np.min(values))
            result_dict["%s_max" % name].append(np.max(values))
    return pd.DataFrame(result_dict)


def evaluate_model_config(
        config,
        allele_datasets,

        min_samples_per_allele=5,
        cv_folds=5,
        learning_rate=0.001):
    print("===")
    print(config)
    if config.embedding_size:
        model = make_embedding_network(
            peptide_length=9,
            embedding_input_dim=20,
            embedding_output_dim=config.embedding_size,
            layer_sizes=[config.hidden_layer_size],
            activation=config.activation,
            init=config.init,
            loss=config.loss,
            dropout_probability=config.dropout_probability,
            learning_rate=learning_rate)
    else:
        model = make_hotshot_network(
            peptide_length=9,
            layer_sizes=[config.hidden_layer_size],
            activation=config.activation,
            init=config.init,
            loss=config.loss,
            dropout_probability=config.dropout_probability,
            learning_rate=learning_rate)
    return leave_out_allele_cross_validation(
        model,
        allele_datasets=allele_datasets,
        max_ic50=config.max_ic50,
        binary_encoding=config.embedding_size == 0,
        n_pretrain_epochs=config.n_pretrain_epochs,
        n_training_epochs=config.n_epochs,
        min_samples_per_allele=min_samples_per_allele,
        cv_folds=cv_folds,
        minibatch_size=config.minibatch_size)
