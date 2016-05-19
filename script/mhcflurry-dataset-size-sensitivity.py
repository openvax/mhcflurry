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
from mhcflurry.args import (
    add_imputation_argument_to_parser,
    add_hyperparameter_arguments_to_parser,
    add_training_arguments_to_parser,
    imputer_from_args,
    predictor_from_args,
)

parser = ArgumentParser()

parser.add_argument(
    "--training-csv",
    default="bdata.2009.mhci.public.1.txt")

parser.add_argument(
    "--allele",
    default="A0201")


parser.add_argument(
    "--repeat",
    type=int,
    default=1,
    help="How many times to train model for same dataset size")

parser.add_argument(
    "--number-dataset-sizes",
    type=int,
    default=10)

parser.add_argument(
    "--min-training-samples",
    type=int,
    default=20)


parser.add_argument(
    "--max-training-samples",
    type=int,
    default=2000)

"""
parser.add_argument(
    "--remove-similar-peptides-from-test-data",
    action="store_true",
    default=False,
    help=(
        "Use a 4 letter reduced amino acid alphabet to identify and "
        "remove correlated peptides from the test data."))
"""

add_imputation_argument_to_parser(parser)
add_hyperparameter_arguments_to_parser(parser)
add_training_arguments_to_parser(parser)

def subsample_performance(
        dataset,
        allele,
        model_fn,
        imputer=None,
        min_training_samples=20,
        max_training_samples=3000,
        n_subsample_sizes=10,
        n_repeats_per_size=1,
        n_training_epochs=200,
        n_random_negative_samples=100,
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

    log_sample_sizes = np.linspace(log_min_samples, log_max_samples, num=n_subsample_sizes)
    sample_sizes = np.exp(log_sample_sizes).astype(int) + 1

    for i, n_train in enumerate(sample_sizes):
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
            print("=== #%d/%d: Training model for %s with sample_size = %d/%d" % (
                i + 1,
                len(sample_sizes),
                allele,
                n_train,
                n_total))

            # pick a fraction on a log-scale from the minimum to maximum number
            # of samples
            model = model_fn()
            model.fit_dataset(
                dataset_train,
                dataset_imputed,
                n_training_epochs=n_training_epochs,
                n_random_negative_samples=n_random_negative_samples)
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
        return predictor_from_args(allele_name=args.allele, args=args)

    xs, aucs, f1s = subsample_performance(
        dataset=dataset,
        allele=args.allele,
        imputer=imputer,
        model_fn=make_model,
        n_repeats_per_size=args.repeat,
        n_training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        min_training_samples=args.min_training_samples,
        max_training_samples=args.max_training_samples,
        n_subsample_sizes=args.number_dataset_sizes,
        n_random_negative_samples=args.random_negative_samples)

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
        filename = "%s-%s-vs-nsamples-hidden-%s-activation-%s-impute-%s.png" % (
            args.allele,
            name,
            args.hidden_layer_size,
            args.activation,
            args.imputation_method)
        figure.savefig(filename)
