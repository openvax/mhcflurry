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
Plot AUC and F1 score of predictors as a function of dataset size
"""

from argparse import ArgumentParser

import numpy as np
import sklearn
import sklearn.metrics
import seaborn

from mhcflurry.dataset import Dataset
from mhcflurry.class1_binding_predictor import Class1BindingPredictor

from dataset_paths import PETERS2009_CSV_PATH

parser = ArgumentParser()

parser.add_argument(
    "--training-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--allele",
    default="A0201")

parser.add_argument(
    "--max-ic50",
    type=float,
    default=50000.0)

parser.add_argument(
    "--hidden-layer-size",
    type=int,
    default=10,
    help="Hidden layer size for neural network, if 0 use linear regression")

parser.add_argument(
    "--embedding-dim",
    type=int,
    default=50,
    help="Number of dimensions for vector embedding of amino acids")

parser.add_argument(
    "--activation",
    default="tanh")

parser.add_argument(
    "--training-epochs",
    type=int,
    default=100)

parser.add_argument(
    "--minibatch-size",
    type=int,
    default=128)

parser.add_argument(
    "--repeat",
    type=int,
    default=10,
    help="How many times to train model for same dataset size")


def subsample_performance(
        dataset,
        allele,
        pretraining=False,
        model_fn=None,
        n_training_samples=[
            25,
            50,
            100,
            200,
            300,
            400,
            500,
            600,
            800,
            1000,
            1200,
            1400,
            1600,
            1800,
            2000],
        niters=10,
        n_training_epochs=200,
        batch_size=32):
    n_total = len(dataset)

    xs = []
    aucs = []
    f1s = []

    for iternum in range(niters):

        model = model_fn()
        initial_weights = model.get_weights()

        mask = np.random.rand(n) > fraction_test
        X_train = X[mask]
        X_test = X[~mask]
        Y_train = Y[mask]
        Y_test = Y[~mask]
        n_train = len(Y_train)
        train_indices = np.arange(len(Y_train))
        np.random.shuffle(train_indices)
        for i, fraction in enumerate(fractions):

            n_fraction = int(n_train * fraction)
            subset_indices = train_indices[:n_fraction]
            X_subset = X_train[subset_indices]
            Y_subset = Y_train[subset_indices]
            if model_fn is None:
                model.fit(X_subset, Y_subset)
            else:
                model.set_weights(initial_weights)
                model.fit(
                    X_subset,
                    Y_subset,
                    verbose=0,
                    nb_epoch=nb_epoch,
                    batch_size=batch_size)
            pred = model.predict(X_test)
            true_ic50 = max_ic50 ** (1 - Y_test)
            true_label = true_ic50 <= 500
            auc = sklearn.metrics.roc_auc_score(true_label, pred)
            xs.append(n_fraction)
            aucs.append(auc)
            pred_ic50 = max_ic50 ** (1 - pred)
            pred_label = pred_ic50 <= 500
            f1 = sklearn.metrics.f1_score(true_label, pred_label)
            print("Fraction=%0.2f, n=%d, AUC=%0.4f, F1=%0.4f" % (fraction, n_fraction, auc, f1))
            f1s.append(f1)
    return xs, aucs, f1s

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = Dataset.load_csv(args.training_csv)
    X = dataset.X
    Y = dataset.Y
    print("Total # of samples for %s: %d" % (args.allele, len(Y)))
    if args.hidden_layer_size == 0:
        hidden_layer_sizes = []
    else:
        hidden_layer_sizes = [args.hidden_layer_size]

    def make_model():
        return Class1BindingPredictor.from_hyperparameters(
            layer_sizes=[args.hidden_layer_size],
            activation=args.activation,
            embedding_dim=args.embedding_dim)

    xs, aucs, f1s = subsample_performance(
        X=X,
        Y=Y,
        model_fn=model_fn,
        max_ic50=args.max_ic50,
        fractions=np.arange(0.01, 1, 0.03),
        niters=args.repeat,
        nb_epoch=args.training_epochs,
        batch_size=args.minibatch_size)
    for (name, values) in [("AUC", aucs), ("F1", f1s)]:
        figure = seaborn.plt.figure(figsize=(10, 8))
        ax = figure.add_axes()
        seaborn.regplot(
            x=np.array(xs).astype(float),
            y=np.array(values),
            logx=True,
            x_jitter=1,
            fit_reg=False,
            color="red",
            scatter_kws=dict(alpha=0.5, s=50))
        seaborn.plt.xlabel("# samples (subset of %s)" % args.allele)
        seaborn.plt.ylabel(name)
        if args.hidden_layer_size:
            filename = "%s-%s-vs-nsamples-hidden-%s-activation-%s.png" % (
                args.allele,
                name,
                args.hidden_layer_size,
                args.activation)
        else:
            filename = "%s-%s-vs-nsamples-linear.png" % (
                args.allele,
                name)
        figure.savefig(filename)
