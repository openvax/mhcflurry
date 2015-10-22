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

from os import listdir
from os.path import join
from argparse import ArgumentParser

import pandas as pd

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
    default=150,
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
    print(config)
    print(model)
    assert False

    # pre-train
    model.fit(None, None)

    old_weights = model.get_weights()
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
        if args.predictor_name in columns:
            raise ValueError("Column '%s' already exists in %s" % (
                args.predictor_name, filepath))

        model.set_weights(old_weights)
        allele_dataset = None
        model.fit(allele_dataset.X, allele_dataset.Y)

        X_test = encode(df[args.peptide_sequence_column_name])
        Y_pred = model.predict(X_test)
        Y_pred_ic50 = args.max_ic50 ** (1.0 - Y_pred)
