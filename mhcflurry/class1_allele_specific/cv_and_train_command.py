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
 * Parallelized with joblib

Note:

The joblib-based parallelization is primary intended to be used with an
alternative joblib backend such as dask-distributed that supports
multi-node parallelization. Theano in particular seems to have deadlocks
when running with single-node parallelization.

Also, when using the multiprocessing backend for joblib (the default),
the 'fork' mode causes a library we use to hang. We have to instead use
the 'spawn' or 'forkserver' modes. See:
https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries
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
import joblib

from ..dataset import Dataset
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
    "--joblib-num-jobs",
    type=int,
    default=1,
    metavar="N",
    help="Number of joblib workers. Set to -1 to use as many jobs as cores. "
    "Default: %(default)s")

parser.add_argument(
    "--joblib-pre-dispatch",
    metavar="STRING",
    default='2*n_jobs',
    help="Tasks to initially dispatch to joblib. Default: %(default)s")

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


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    if not args.quiet:
        logging.basicConfig(level="INFO")
    if args.verbose:
        logging.basicConfig(level="DEBUG")
    if args.dask_scheduler:
        import distributed.joblib  # for side effects
        backend = joblib.parallel_backend(
            'distributed',
            scheduler_host=args.dask_scheduler)
        with backend:
            active_backend = joblib.parallel.get_active_backend()[0]
            logging.info(
                "Running with dask scheduler: %s [%d cores]" % (
                    args.dask_scheduler,
                    active_backend.effective_n_jobs()))

            go(args)
    else:
        go(args)


def go(args):
    model_architectures = json.loads(args.model_architectures.read())
    logging.info("Read %d model architectures" % len(model_architectures))
    if args.max_models:
        model_architectures = model_architectures[:args.max_models]
        logging.info(
            "Subselected to %d model architectures" % len(model_architectures))

    train_data = Dataset.from_csv(args.train_data)
    logging.info("Loaded training dataset: %s" % train_data)

    test_data = None
    if args.test_data:
        test_data = Dataset.from_csv(args.test_data)
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
        alleles=args.alleles,
        n_jobs=args.joblib_num_jobs,
        pre_dispatch=args.joblib_pre_dispatch,
        verbose=1 if not args.quiet else 0)

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
        folds_per_task=args.cv_folds_per_task,
        n_jobs=args.joblib_num_jobs,
        verbose=1 if not args.quiet else 0,
        pre_dispatch=args.joblib_pre_dispatch)
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
    imputation_tasks = []
    for (allele_num, allele) in enumerate(cv_results.allele.unique()):
        best_index = best_architectures_by_allele[allele]
        architecture = model_architectures[best_index]
        train_models.append(architecture)
        logging.info(
            "Allele: %s best architecture is index %d: %s" %
            (allele, best_index, architecture))

        if architecture['impute']:
            imputation_task = joblib.delayed(impute_and_select_allele)(
                train_data,
                imputer=imputer,
                allele=allele,
                **impute_kwargs)
            imputation_tasks.append(imputation_task)
        else:
            imputation_task = None

        test_data_this_allele = None
        if test_data is not None:
            test_data_this_allele = test_data.get_allele(allele)
        fold = AlleleSpecificTrainTestFold(
            allele=allele,
            train=train_data.get_allele(allele),

            # Here we set imputed_train to the imputation *task* if
            # imputation was used on this fold. We set this to the actual
            # imputed training dataset a few lines farther down. This
            # complexity is because we want to be able to parallelize
            # the imputations so we have to queue up the tasks first.
            # If we are not doing imputation then the imputation_task
            # is None.
            imputed_train=imputation_task,
            test=test_data_this_allele)
        train_folds.append(fold)

    if imputation_tasks:
        logging.info(
            "Waiting for %d full-data imputation tasks to complete"
            % len(imputation_tasks))
        imputation_results = joblib.Parallel(
            n_jobs=args.joblib_num_jobs,
            verbose=1 if not args.quiet else 0,
            pre_dispatch=args.joblib_pre_dispatch)(imputation_tasks)

        train_folds = [
            train_fold._replace(
                # Now we replace imputed_train with the actual imputed
                # dataset.
                imputed_train=imputation_results.pop(0)
                if (train_fold.imputed_train is not None) else None)
            for train_fold in train_folds
        ]
        assert not imputation_results
        del imputation_tasks

    logging.info("Training %d production models" % len(train_folds))
    start = time.time()
    train_results = train_across_models_and_folds(
        train_folds,
        train_models,
        cartesian_product_of_folds_and_models=False,
        return_predictors=args.out_models_dir is not None,
        n_jobs=args.joblib_num_jobs,
        verbose=1 if not args.quiet else 0,
        pre_dispatch=args.joblib_pre_dispatch)
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
