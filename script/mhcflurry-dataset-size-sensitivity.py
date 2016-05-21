#!/usr/bin/env python3
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
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.metrics
import seaborn
from matplotlib import pyplot


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
    default=3,
    help="How many times to train model for same dataset size")

parser.add_argument(
    "--number-dataset-sizes",
    type=int,
    default=20)

parser.add_argument(
    "--min-training-samples",
    type=int,
    default=5)

parser.add_argument(
    "--max-training-samples",
    type=int,
    default=500)

parser.add_argument(
    "--min-observations-per-peptide",
    type=int,
    default=2)

parser.add_argument(
    "--sample-censored-affinities",
    default=False,
    action="store_true")

parser.add_argument(
    "--load-existing-data",
    action="store_true",
    default=False,
    help="Skip computationally intensive data generation by loading from existing CSV")

parser.add_argument(
    "--seaborn-lmplot",
    action="store_true",
    default=False)

parser.add_argument(
    "--pretraining-weight-decay",
    choices=("exponential", "quadratic", "linear"),
    default="quadratic",
    help="Rate at which weight of imputed samples decays")
"""
parser.add_argument(
    "--remove-similar-peptides-from-test-data",
    action="store_true",
    default=False,
    help=(
        "Use a 4 letter reduced amino acid alphabet to identify and "
        "remove correlated peptides from the test data."))
"""

add_imputation_argument_to_parser(parser, default="mice")
add_hyperparameter_arguments_to_parser(parser)
add_training_arguments_to_parser(parser)

def stratify_by_binder_label(row):
    return row["affinity"] <= 500


def subsample_performance(
        dataset,
        allele,
        model_fn,
        imputer=None,
        min_training_samples=20,
        max_training_samples=3000,
        min_observations_per_peptide=2,
        n_subsample_sizes=10,
        n_repeats_per_size=1,
        n_training_epochs=200,
        n_random_negative_samples=100,
        batch_size=32,
        pretrain_weight_decay_fn=lambda t: np.exp(-t),
        sample_censored_affinities=False):

    dataset_allele = dataset.get_allele(allele)
    n_total = len(dataset_allele)

    # subsampled training set should be at most 2/3 of the total data
    max_training_samples = min((2 * n_total) // 3, max_training_samples)

    results = OrderedDict([
        ("num_samples", []),
        ("idx", []),
        ("auc", []),
        ("f1", []),
        ("tau", []),
        ("impute", []),
    ])
    sample_sizes = np.exp(np.linspace(
        np.log(min_training_samples),
        np.log(max_training_samples),
        num=n_subsample_sizes)).astype(int) + 1

    i = 0
    for n_train in sample_sizes:
        for _ in range(n_repeats_per_size):
            i += 1
            if imputer is None:
                dataset_train, dataset_test = dataset_allele.random_split(
                    n_train, stratify_fn=stratify_by_binder_label)
                dataset_imputed = None
            else:
                dataset_train, dataset_imputed, dataset_test = \
                    dataset.split_allele_randomly_and_impute_training_set(
                        allele=allele,
                        n_training_samples=n_train,
                        imputation_method=imputer,
                        min_observations_per_peptide=min_observations_per_peptide,
                        min_observations_per_allele=1,
                        stratify_fn=stratify_by_binder_label)
            print("=== #%d/%d: Training model for %s with sample_size = %d/%d" % (
                i + 1,
                len(sample_sizes) * n_repeats_per_size,
                allele,
                n_train,
                n_total))
            print("-- Train", dataset_train)
            print("-- Imputed", dataset_imputed)
            print("-- Test", dataset_test)
            print("-- %d/%d training samples strong binders" % (
                (dataset_train.affinities <= 500).sum(),
                len(dataset_train)))
            print("-- %d/%d test samples strong binders" % (
                (dataset_test.affinities <= 500).sum(),
                len(dataset_test)))
            true_ic50 = dataset_test.affinities
            true_label = true_ic50 <= 500

            # pick a fraction on a log-scale from the minimum to maximum number
            # of samples
            model = model_fn()
            initial_weights = model.get_weights()

            print("-- Fitting model without imputation/pretraining")
            model.fit_dataset(
                dataset_train,
                n_training_epochs=n_training_epochs,
                n_random_negative_samples=n_random_negative_samples)
            pred_ic50 = model.predict(dataset_test.peptides)
            auc = sklearn.metrics.roc_auc_score(true_label, -pred_ic50)
            tau = scipy.stats.kendalltau(true_ic50, pred_ic50).correlation
            f1 = sklearn.metrics.f1_score(true_label, pred_ic50 <= 500)
            print("%s (impute=none) n=%d, AUC=%0.4f, F1=%0.4f, tau=%0.4f" % (
                allele, n_train, auc, f1, tau))
            results["num_samples"].append(n_train)
            results["idx"].append(i)
            results["auc"].append(auc)
            results["f1"].append(f1)
            results["tau"].append(tau)
            results["impute"].append(False)

            if dataset_imputed is None:
                continue

            print("-- Fitting model WITH imputation/pretraining")
            model.set_weights(initial_weights)
            model.fit_dataset(
                dataset_train,
                dataset_imputed,
                n_training_epochs=n_training_epochs,
                n_random_negative_samples=n_random_negative_samples,
                sample_censored_affinities=sample_censored_affinities)
            pred_ic50 = model.predict(dataset_test.peptides)
            auc = sklearn.metrics.roc_auc_score(true_label, -pred_ic50)
            tau = scipy.stats.kendalltau(true_ic50, pred_ic50).correlation

            f1 = sklearn.metrics.f1_score(true_label, pred_ic50 <= 500)
            print("%s (impute=%s) n=%d, AUC=%0.4f, F1=%0.4f, tau=%0.4f" % (
                allele, imputer, n_train, auc, f1, tau))
            results["num_samples"].append(n_train)
            results["idx"].append(i)
            results["auc"].append(auc)
            results["f1"].append(f1)
            results["tau"].append(tau)
            results["impute"].append(True)

    return pd.DataFrame(results)

if __name__ == "__main__":
    args = parser.parse_args()

    base_filename = \
        ("%s-vs-nsamples-hidden-%s-activation-%s"
         "-impute-%s-epochs-%d-embedding-%d-pretrain-%s") % (
            args.allele,
            args.hidden_layer_size,
            args.activation,
            args.imputation_method,
            args.training_epochs,
            args.embedding_size,
            args.pretraining_weight_decay)
    csv_filename = base_filename + ".csv"

    if args.load_existing_data:
        results_df = pd.read_csv(csv_filename)
    else:
        dataset = Dataset.from_csv(args.training_csv)
        imputer = imputer_from_args(args)

        def make_model():
            return predictor_from_args(allele_name=args.allele, args=args)

        if args.pretraining_weight_decay == "exponential":
            def pretrain_weight_decay_fn(t):
                return np.exp(-t)

        elif args.pretraining_weight_decay == "quadratic":
            def pretrain_weight_decay_fn(t):
                return 1.0 / (t + 1) ** 2.0

        elif args.pretraining_weight_decay == "linear":
            def pretrain_weight_decay_fn(t):
                return 1.0 / (t + 1)

        else:
            raise ValueError("Invalid weight decay schedule: '%s'" % (
                args.pretraining_weight_decay))

        results_df = subsample_performance(
            dataset=dataset,
            allele=args.allele,
            imputer=imputer,
            model_fn=make_model,
            n_repeats_per_size=args.repeat,
            n_training_epochs=args.training_epochs,
            batch_size=args.batch_size,
            pretrain_weight_decay_fn=pretrain_weight_decay_fn,
            min_training_samples=args.min_training_samples,
            max_training_samples=args.max_training_samples,
            min_observations_per_peptide=args.min_observations_per_peptide,
            n_subsample_sizes=args.number_dataset_sizes,
            n_random_negative_samples=args.random_negative_samples,
            sample_censored_affinities=args.sample_censored_affinities)
        print("Writing CSV to %s" % csv_filename)
        results_df.to_csv(csv_filename)

    print(results_df)

    metrics = ["auc", "f1", "tau"]

    titles = {
        "tau": "Kendall's $\\tau$",
        "auc": "AUC",
        "f1": "$F_1$ score"
    }
    pyplot.figure(figsize=(6.5, 3.5))
    seaborn.set_style("whitegrid")

    for (j, score_name) in enumerate(metrics):
        ax = pyplot.subplot2grid((1, 4), (0, j))
        groups = results_df.groupby(["num_samples", "impute"])
        groups_score = groups[score_name].mean().to_frame().reset_index()
        groups_score["std_error"] = \
            groups[score_name].std().to_frame().reset_index()[score_name]

        for impute in [True, False]:
            sub = groups_score[groups_score.impute == impute]
            color = seaborn.get_color_cycle()[0] if impute else seaborn.get_color_cycle()[1]
            pyplot.errorbar(
                x=sub.num_samples.values,
                y=sub[score_name].values,
                yerr=sub.std_error.values,
                label=("with" if impute else "without") + " imputation",
                color=color)
        if j == 1:
            pyplot.xlabel("Training set size")
        pyplot.xscale("log")
        pyplot.title(titles[score_name])

        if score_name == "auc":
            pyplot.ylim(ymin=0.5, ymax=1.0)
        if score_name == "f1":
            pyplot.ylim(ymin=0, ymax=1)
        if score_name == "tau":
            pyplot.ylim(ymin=0, ymax=0.6)
            pyplot.yticks(np.arange(0, 0.61, 0.15))

    pyplot.legend(
        bbox_to_anchor=(1.1, 1),
        loc=2,
        borderaxespad=0.,
        fancybox=True,
        frameon=True,
        fontsize="small")

    pyplot.tight_layout()
    # Put the legend out of the figure
    image_filename = base_filename + ".png"
    print("Writing PNG to %s" % image_filename)
    pyplot.savefig(image_filename)

    pdf_filename = base_filename + ".pdf"
    print("Writing PDF to %s" % pdf_filename)
    pyplot.savefig(pdf_filename)
