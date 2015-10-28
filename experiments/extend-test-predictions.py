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
from mhcflurry.data_helpers import load_data, index_encoding, hotshot_encoding
from mhcflurry.common import normalize_allele_name, expand_9mer_peptides

from dataset_paths import PETERS2009_CSV_PATH
from model_configs import ModelConfig
from model_selection_helpers import make_model
"""
namedtuple(
    "ModelConfig",
    [
        "embedding_size",
        "hidden_layer_size",
        "activation",
        "loss",
        "init",
        "n_pretrain_epochs",
        "n_epochs",
        "dropout_probability",
        "max_ic50",
        "minibatch_size",
        "learning_rate",
        "optimizer",
    ])
"""


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
    default=400,
    type=int,
    help="Hidden layer size")


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

    model = make_model(config)
    binary_encoding = (args.embedding_size == 0)

    training_datasets, _ = load_data(
        filename=args.training_csv,
        peptide_length=9,
        max_ic50=args.max_ic50,
        binary_encoding=binary_encoding)

    X_all = np.vstack([dataset.X for dataset in training_datasets.values()])
    Y_all = np.concatenate([
        dataset.Y
        for dataset in training_datasets.values()
    ])
    model.fit(
        X_all,
        Y_all,
        nb_epoch=args.pretrain_epochs,
        batch_size=args.minibatch_size,
        shuffle=True)
    old_weights = model.get_weights()

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

        model.set_weights(old_weights)
        allele_dataset = training_datasets[allele_name]
        X_train = allele_dataset.X
        Y_train = allele_dataset.Y

        model.fit(
            X_train,
            Y_train,
            nb_epoch=args.training_epochs,
            batch_size=args.minibatch_size,
            shuffle=True)

        predictions = {}
        for length, equal_length_sequences in groupby(
                peptide_sequences,
                lambda seq: len(seq)):
            for peptide in equal_length_sequences:
                expanded_peptides = expand_9mer_peptides([peptide], length=length)
                if binary_encoding:
                    X_test = hotshot_encoding(
                        expanded_peptides,
                        peptide_length=9)
                    # collapse 3D input into 2D matrix
                    X_test = X_test.reshape((X_test.shape[0], 9 * 20))
                else:
                    X_test = index_encoding(
                        expanded_peptides,
                        peptide_length=9)

                Y_pred = model.predict(X_test)

                assert len(X_test) == len(Y_pred)
                Y_pred_mean = np.mean(Y_pred)
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
