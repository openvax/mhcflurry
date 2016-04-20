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
Given
1) A training set of binding results
2) A collection of allele-specific CSV files containing measurements and
predictions on a separate test set

...train a mhcflurry predictor on the training set and extend each allele
of the test set with mchflurry predictions (writing results to a new directory)
"""

from os import listdir, makedirs
from os.path import join, exists
from argparse import ArgumentParser
from itertools import groupby

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from mhcflurry.data import load_allele_datasets
from mhcflurry.peptide_encoding import (
    fixed_length_index_encoding,
    indices_to_hotshot_encoding
)
from mhcflurry.common import normalize_allele_name

from dataset_paths import PETERS2009_CSV_PATH
from model_configs import ModelConfig
from model_selection_helpers import make_model

parser = ArgumentParser()

parser.add_argument(
    "--training-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--input-dir",
    help="Directory which contains one CSV file per allele",
    required=True)

parser.add_argument(
    "--output-dir",
    help="Directory which contains one CSV file per allele",
    required=True)

parser.add_argument(
    "--peptide-sequence-column-name",
    default="sequence")

parser.add_argument(
    "--predictor-name",
    default="mhcflurry")

parser.add_argument(
    "--pretrain-epochs",
    default=10,
    type=int,
    help="Number of pre-training epochs which use all allele data combined")

parser.add_argument(
    "--training-epochs",
    default=200,
    type=int,
    help="Number of passes over the dataset to perform during model fitting")

parser.add_argument(
    "--dropout",
    default=0.5,
    type=float,
    help="Degree of dropout regularization to try in hyperparameter search")

parser.add_argument(
    "--minibatch-size",
    default=256,
    type=int,
    help="How many samples to use in stochastic gradient estimation")

parser.add_argument(
    "--embedding-size",
    default=64,
    type=int,
    help="Size of vector embedding dimension")

parser.add_argument(
    "--learning-rate",
    default=0.001,
    type=float,
    help="Learning rate for RMSprop")

parser.add_argument(
    "--hidden-layer-size",
    default=200,
    type=int,
    help="Hidden layer size, default = multiple of 25 nearest n_samples/20")


parser.add_argument(
    "--max-ic50",
    default=20000.0,
    type=float,
    help="Maximum predicted IC50 values")


parser.add_argument(
    "--init",
    default="glorot_uniform",
    help="Initialization methods")

parser.add_argument(
    "--activation",
    default="tanh",
    help="Activation functions")

parser.add_argument(
    "--optimizer",
    default="rmsprop",
    help="Optimization methods")

parser.add_argument("--ensemble-size", type=int, default=5)

if __name__ == "__main__":
    args = parser.parse_args()

    config = ModelConfig(
        embedding_size=args.embedding_size,
        hidden_layer_size=args.hidden_layer_size,
        activation=args.activation,
        loss="mse",
        init=args.init,
        n_pretrain_epochs=args.pretrain_epochs,
        n_epochs=args.training_epochs,
        dropout_probability=args.dropout,
        max_ic50=args.max_ic50,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer)

    models = [make_model(config) for _ in range(args.ensemble_size)]

    binary_encoding = (args.embedding_size == 0)

    training_datasets = load_allele_datasets(
        filename=args.training_csv,
        peptide_length=9,
        max_ic50=args.max_ic50)

    X_all = np.vstack([dataset.X_index for dataset in training_datasets.values()])
    Y_all = np.concatenate([
        dataset.Y
        for dataset in training_datasets.values()
    ])

    for model in models:
        model.fit(
            X_all,
            Y_all,
            nb_epoch=args.pretrain_epochs,
            batch_size=args.minibatch_size,
            shuffle=True)
    old_weights = [model.get_weights() for model in models]

    if not exists(args.output_dir):
        makedirs(args.output_dir)
    for filename in listdir(args.input_dir):
        filepath = join(args.input_dir, filename)
        parts = filename.split(".")
        if len(parts) != 2:
            print("Skipping %s" % filepath)
            continue
        allele_name, ext = parts
        if ext != "csv":
            print("Skipping %s, only reading CSV files" % filepath)
            continue

        allele_name = normalize_allele_name(allele_name)
        if allele_name not in training_datasets:
            print("Skipping %s because allele %s not in training data" % (
                filepath,
                allele_name))
            continue

        print("Loading %s" % filepath)
        df = pd.read_csv(filepath)

        columns = set(df.columns)
        if args.predictor_name in columns:
            raise ValueError("Column '%s' already exists in %s" % (
                args.predictor_name, filepath))

        peptide_sequences = list(df["sequence"])
        true_ic50 = list(df["meas"])

        for model_i, old_weights_i in zip(models, old_weights):
            model_i.set_weights(old_weights_i)

        allele_dataset = training_datasets[allele_name]
        X_train = allele_dataset.X
        Y_train = allele_dataset.Y
        training_epochs = args.training_epochs
        if not training_epochs:
            training_epochs = max(1, int(10 ** 6 / len(Y_train)))

        for i, (cv_train_indices, cv_test_indices) in enumerate(StratifiedKFold(
                y=(Y_train <= 500),
                n_folds=args.ensemble_size,
                shuffle=True)):
            for epoch in range(args.training_epochs):
                models[i].fit(
                    X_train[cv_train_indices, :],
                    Y_train[cv_train_indices],
                    nb_epoch=1,
                    batch_size=args.minibatch_size,
                    verbose=0)
                cv_train_pred = models[i].predict(X_train[cv_train_indices, :])
                cv_train_pred = cv_train_pred.flatten()
                cv_train_mse = ((
                    cv_train_pred - Y_train[cv_train_indices]) ** 2).mean()
                cv_test_pred = models[i].predict(X_train[cv_test_indices, :])
                cv_test_pred = cv_test_pred.flatten()
                cv_test_mse = ((
                    cv_test_pred - Y_train[cv_test_indices]) ** 2).mean()

                print("Model #%d epoch #%d train MSE=%0.4f test MSE=%0.4f" % (
                    i + 1,
                    epoch + 1,
                    cv_train_mse,
                    cv_test_mse,
                ))

        predictions = {}
        for length, equal_length_sequences in groupby(
                peptide_sequences,
                lambda seq: len(seq)):
            for peptide in equal_length_sequences:
                n_indices = 21
                X_test_index = fixed_length_index_encoding(
                    peptides=[peptide],
                    desired_length=length,
                    allow_unknown_amino_acids=True)
                if binary_encoding:
                    X_test = indices_to_hotshot_encoding(X_test_index, n_indices=n_indices)
                else:
                    X_test = X_test_index
                Y_preds = [model.predict(X_test) for model in models]
                Y_pred_mean = np.mean(Y_preds)
                Y_pred_ic50 = args.max_ic50 ** (1.0 - Y_pred_mean)
                predictions[peptide] = Y_pred_ic50

        df[args.predictor_name] = [
            predictions[peptide]
            for peptide in peptide_sequences
        ]

        pos = df["meas"] <= 500
        pred = df[args.predictor_name] <= 500
        tp = (pred & pos).sum()
        fp = (pred & ~pos).sum()
        tn = (~pred & ~pos).sum()
        fn = (~pred & pos).sum()
        assert (tp + fp + tn + fn) == len(pos), "Expected %d but got %d" % (
            len(pos),
            (tp + fp + tn + fn))
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print("-- %s: tp=%d fp=%d tn=%d fn=%d P=%0.4f R=%0.4f F1=%0.4f" % (
            filename, tp, fp, tn, fn, precision, recall, f1))
        output_path = join(args.output_dir, filename)
        df.to_csv(output_path, index=False)
