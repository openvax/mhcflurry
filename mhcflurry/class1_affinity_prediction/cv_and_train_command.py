# Copyright (c) 2016. Mount Sinai School of Medicine
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
'''
Class1 allele-specific cross validation and training script.

What it does:
 * Run cross validation on a dataset over the specified model architectures
 * Select the best architecture for each allele
 * Re-train the best architecture on the full data for that allele
 * Test "production" predictors on a held-out test set if available

Features:
 * Supports imputation as a hyperparameter that can be searched over
 * Parallelized with concurrent.futures

Note:

The parallelization is primary intended to be used with an
alternative concurrent.futures Executor such as dask-distributed that supports
multi-node parallelization. Theano in particular seems to have deadlocks
when running with single-node parallelization.
'''
from __future__ import (
    print_function,
    division,
    absolute_import,
)
import sys
import argparse
import json
import logging
import time
import os
import socket
import hashlib
import pickle

import numpy

from .. import parallelism
from ..affinity_measurement_dataset import AffinityMeasurementDataset
from ..imputation_helpers import imputer_from_name
from .cross_validation import cross_validation_folds
from .train import (
    impute_and_select_allele,
    train_across_models_and_folds,
    AlleleSpecificTrainTestFold)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument(
    "--train-data",
    metavar="X.csv",
    required=True,
    help="Training data")

parser.add_argument(
    "--test-data",
    metavar="X.csv",
    help="Optional test data")

parser.add_argument(
    "--model-architectures",
    metavar="X.json",
    type=argparse.FileType('r'),
    required=True,
    help="JSON file giving model architectures to assess in cross validation."
    " Can be - to read from stdin")

parser.add_argument(
    "--imputer-description",
    metavar="X.json",
    type=argparse.FileType('r'),
    help="JSON. Can be - to read from stdin")

parser.add_argument(
    "--alleles",
    metavar="ALLELE",
    nargs="+",
    default=None,
    help="Use only the specified alleles")

parser.add_argument(
    "--out-cv-results",
    metavar="X.csv",
    help="Write cross validation results to the given file")

parser.add_argument(
    "--out-production-results",
    metavar="X.csv",
    help="Write production model information to the given file")

parser.add_argument(
    "--out-models-dir",
    metavar="DIR",
    help="Write production models to files in this dir")

parser.add_argument(
    "--max-models",
    type=int,
    metavar="N",
    help="Use only the first N models")

parser.add_argument(
    "--cv-num-folds",
    type=int,
    default=3,
    metavar="N",
    help="Number of cross validation folds. Default: %(default)s")

parser.add_argument(
    "--cv-folds-per-task",
    type=int,
    default=10,
    metavar="N",
    help="When parallelizing cross validation, each task trains one model "
    "architecture on N folds. Set to 1 for maximum potential parallelism. "
    "This is less efficient if you have limited workers, however, since "
    "the model must get compiled for each task. Default: %(default)s.")

parser.add_argument(
    "--dask-scheduler",
    metavar="HOST:PORT",
    help="Host and port of dask distributed scheduler")

parser.add_argument(
    "--num-local-processes",
    metavar="N",
    type=int,
    help="Processes (exclusive with --dask-scheduler and --num-local-threads)")

parser.add_argument(
    "--num-local-threads",
    metavar="N",
    type=int,
    default=1,
    help="Threads (exclusive with --dask-scheduler and --num-local-processes)")

parser.add_argument(
    "--min-samples-per-allele",
    default=100,
    metavar="N",
    help="Don't train predictors for alleles with fewer than N samples. "
    "Set to 0 to disable filtering. Default: %(default)s",
    type=int)

parser.add_argument(
    "--quiet",
    action="store_true",
    default=False,
    help="Output less info")

parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Output more info")

try:
    import kubeface
    kubeface.Client.add_args(parser)
except ImportError:
    logging.error("Kubeface support disabled, not installed.")


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    if args.verbose:
        logging.root.setLevel(level="DEBUG")
    elif not args.quiet:
        logging.root.setLevel(level="INFO")

    logging.info("Running with arguments: %s" % args)

    # Set parallel backend
    if args.dask_scheduler:
        backend = parallelism.DaskDistributedParallelBackend(
            args.dask_scheduler)
    elif hasattr(args, 'storage_prefix') and args.storage_prefix:
        backend = parallelism.KubefaceParallelBackend(args)
    else:
        if args.num_local_processes:
            backend = parallelism.ConcurrentFuturesParallelBackend(
                args.num_local_processes,
                processes=True)
        else:
            backend = parallelism.ConcurrentFuturesParallelBackend(
                args.num_local_threads,
                processes=False)

    parallelism.set_default_backend(backend)
    logging.info("Using parallel backend: %s" % backend)
    go(args)


def go(args):
    backend = parallelism.get_default_backend()

    model_architectures = json.loads(args.model_architectures.read())
    logging.info("Read %d model architectures" % len(model_architectures))
    if args.max_models:
        model_architectures = model_architectures[:args.max_models]
        logging.info(
            "Subselected to %d model architectures" % len(model_architectures))

    train_data = AffinityMeasurementDataset.from_csv(args.train_data)
    logging.info("Loaded training dataset: %s" % train_data)

    test_data = None
    if args.test_data:
        test_data = AffinityMeasurementDataset.from_csv(args.test_data)
        logging.info("Loaded testing dataset: %s" % test_data)

    if args.min_samples_per_allele:
        train_data = train_data.filter_alleles_by_count(
            args.min_samples_per_allele)
        logging.info(
            "Filtered training dataset to alleles with >= %d observations: %s"
            % (args.min_samples_per_allele, train_data))

    if any(x['impute'] for x in model_architectures):
        if not args.imputer_description:
            parser.error(
                "--imputer-description is required when any models "
                "use imputation")
        imputer_description = json.load(args.imputer_description)
        logging.info("Loaded imputer description: %s" % imputer_description)
        imputer_kwargs_defaults = {
            'min_observations_per_peptide': 2,
            'min_observations_per_allele': 10,
        }
        impute_kwargs = dict(
            (key, imputer_description.pop(key, default))
            for (key, default) in imputer_kwargs_defaults.items())

        imputer = imputer_from_name(**imputer_description)
    else:
        imputer = None
        impute_kwargs = {}

    logging.info(
        "Generating cross validation folds. Imputation: %s" %
        ("yes" if imputer else "no"))
    cv_folds = cross_validation_folds(
        train_data,
        n_folds=args.cv_num_folds,
        imputer=imputer,
        impute_kwargs=impute_kwargs,
        drop_similar_peptides=True,
        alleles=args.alleles)

    logging.info(
        "Training %d model architectures across %d folds = %d models"
        % (
            len(model_architectures),
            len(cv_folds),
            len(model_architectures) * len(cv_folds)))
    start = time.time()
    cv_results = train_across_models_and_folds(
        cv_folds,
        model_architectures,
        folds_per_task=args.cv_folds_per_task)
    logging.info(
        "Completed cross validation in %0.2f seconds" % (time.time() - start))

    cv_results["summary_score"] = (
        cv_results.test_auc.fillna(0) +
        cv_results.test_tau.fillna(0) +
        cv_results.test_f1.fillna(0))

    allele_and_model_to_ranks = {}
    for allele in cv_results.allele.unique():
        model_ranks = (
            cv_results.ix[cv_results.allele == allele]
            .groupby("model_num")
            .summary_score
            .mean()
            .rank(method='first', ascending=False, na_option="top")
            .astype(int))
        allele_and_model_to_ranks[allele] = model_ranks.to_dict()

    cv_results["summary_rank"] = [
        allele_and_model_to_ranks[row.allele][row.model_num]
        for (_, row) in cv_results.iterrows()
    ]

    if args.out_cv_results:
        cv_results.to_csv(args.out_cv_results, index=False)
        print("Wrote: %s" % args.out_cv_results)

    numpy.testing.assert_equal(
        set(cv_results.summary_rank),
        set(1 + numpy.arange(len(model_architectures))))

    best_architectures_by_allele = (
        cv_results.ix[cv_results.summary_rank == 1]
        .set_index("allele")
        .model_num
        .to_dict())

    logging.info("")
    train_folds = []
    train_models = []
    imputation_args_list = []
    best_architectures = []
    for (allele_num, allele) in enumerate(cv_results.allele.unique()):
        best_index = best_architectures_by_allele[allele]
        architecture = model_architectures[best_index]
        best_architectures.append(architecture)
        train_models.append(architecture)
        logging.info(
            "Allele: %s best architecture is index %d: %s" %
            (allele, best_index, architecture))

        if architecture['impute']:
            imputation_args = dict(impute_kwargs)
            imputation_args.update(dict(
                dataset=train_data,
                imputer=imputer,
                allele=allele))
            imputation_args_list.append(imputation_args)

        test_data_this_allele = None
        if test_data is not None:
            test_data_this_allele = test_data.get_allele(allele)
        fold = AlleleSpecificTrainTestFold(
            allele=allele,
            train=train_data.get_allele(allele),
            imputed_train=None,
            test=test_data_this_allele)
        train_folds.append(fold)

    if imputation_args_list:
        imputation_results = list(backend.map(
            lambda kwargs: impute_and_select_allele(**kwargs),
            imputation_args_list))

        new_train_folds = []
        for (best_architecture, train_fold) in zip(
                best_architectures, train_folds):
            imputed_train = None
            if best_architecture['impute']:
                imputed_train = imputation_results.pop(0)
            new_train_folds.append(
                train_fold._replace(imputed_train=imputed_train))
        assert not imputation_results

        train_folds = new_train_folds

    logging.info("Training %d production models" % len(train_folds))
    start = time.time()
    train_results = train_across_models_and_folds(
        train_folds,
        train_models,
        cartesian_product_of_folds_and_models=False,
        return_predictors=args.out_models_dir is not None)
    logging.info(
        "Completed production training in %0.2f seconds"
        % (time.time() - start))

    if args.out_models_dir:
        predictor_names = []
        run_name = (hashlib.sha1(
            ("%s-%f" % (socket.gethostname(), time.time())).encode())
            .hexdigest()[:8])
        for (_, row) in train_results.iterrows():
            predictor_name = "-".join(str(x) for x in [
                row.allele,
                "impute" if row.model_impute else "noimpute",
                "then".join(str(s) for s in row.model_layer_sizes),
                "dropout%g" % row.model_dropout_probability,
                "fracneg%g" % row.model_fraction_negative,
                run_name,
            ]).replace(".", "_")
            predictor_names.append(predictor_name)
            out_path = os.path.join(
                args.out_models_dir, predictor_name + ".pickle")
            with open(out_path, "wb") as fd:
                # Use this protocol so we have Python 2 compatability.
                pickle.dump(row.predictor, fd, protocol=2)
            print("Wrote: %s" % out_path)
        del train_results["predictor"]
        train_results["predictor_name"] = predictor_names

    if args.out_production_results:
        train_results.to_csv(args.out_production_results, index=False)
        print("Wrote: %s" % args.out_production_results)
