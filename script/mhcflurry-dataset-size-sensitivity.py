#!/usr/bin/env python
#
# Copyright (c) 2015-2016. Mount Sinai School of Medicine
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
from mhcflurry.args import add_imputation_argument_to_parser, imputer_from_args

parser = ArgumentParser()

parser.add_argument(
    "--training-csv",
    default="bdata.2009.mhci.public.1.txt")

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

add_imputation_argument_to_parser(parser)


def subsample_performance(
        dataset,
        allele,
        model_fn,
        imputer=None,
        pretraining=False,
        min_training_samples=20,
        max_training_samples=3000,
        n_subsample_sizes=5,
        n_repeats_per_size=3,
        n_training_epochs=200,
        batch_size=32):

    dataset_allele = dataset.get_allele(allele)
    n_total = len(dataset_allele)

    # subsampled training set should be at most 2/3 of the total data
    max_training_samples = min((2 * n_total) // 3, max_training_samples)

    xs = []
    aucs = []
    f1s = []

    log_min_samples = np.log(min_training_samples)
    log_max_samples = np.log(max_training_samples)

    log_sample_sizes = np.linspace(log_min_samples, log_max_samples)
    sample_sizes = np.exp(log_sample_sizes).astype(int)

    for n_train in sample_sizes:
        for _ in range(n_repeats_per_size):
            if imputer is None:
                dataset_train, dataset_test = dataset.random_split(n_train)
                dataset_imputed = None
            else:
                dataset_train, dataset_imputed, dataset_test = \
                    dataset.split_allele_randomly_and_impute_training_set(
                        allele=allele,
                        n_training_samples=n_train,
                        imputation_method=imputer,
                        min_observations_per_peptide=2)
            print("=== Training model for %s with sample_size = %d/%d" % (
                allele,
                n_train,
                n_total))

            # pick a fraction on a log-scale from the minimum to maximum number
            # of samples
            model = model_fn()
            model.fit_dataset(
                dataset_train,
                dataset_imputed,
                n_training_epochs=n_training_epochs)
            pred_ic50 = model.predict(dataset_test.peptides)
            true_ic50 = dataset_test.affinities
            true_label = true_ic50 <= 500
            pred_label = pred_ic50 <= 500
            print("%% Strong Binders True=%f, Predicted=%f" % (
                true_label.mean(),
                pred_label.mean()))
            print("Accuracy = %f" % (true_label == pred_label).mean())
            auc = sklearn.metrics.roc_auc_score(true_label, -pred_ic50)
            xs.append(n_train)
            aucs.append(auc)

            f1 = sklearn.metrics.f1_score(true_label, pred_label)
            print("%s n=%d, AUC=%0.4f, F1=%0.4f" % (allele, n_train, auc, f1))
            f1s.append(f1)
    return xs, aucs, f1s

if __name__ == "__main__":
    args = parser.parse_args()

    dataset = Dataset.from_csv(args.training_csv)
    imputer = imputer_from_args(args)

    def make_model():
        return Class1BindingPredictor.from_hyperparameters(
            layer_sizes=[args.hidden_layer_size] if args.hidden_layer_size > 0 else [],
            activation=args.activation,
            embedding_output_dim=args.embedding_dim)

    xs, aucs, f1s = subsample_performance(
        dataset=dataset,
        allele=args.allele,
        imputer=imputer,
        model_fn=make_model,
        n_repeats_per_size=args.repeat,
        n_training_epochs=args.training_epochs,
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
