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
Ensemble class1 allele-specific model selection and training script.

Procedure
    - N (ensemble size) times:
        - Split full dataset (all alleles) into 50/50 train and test splits.
          Stratify by allele.
        - Perform imputation on train subset.
        - For each allele and architecture, train and test on splits.

    The final predictor is an ensemble of the N best predictors for each
    allele.

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
from ..dataset import Dataset

from .class1_ensemble_multi_allele_predictor import (
    Class1EnsembleMultiAllelePredictor)
from .measurement_collection import MeasurementCollection

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument(
    "--train-data",
    metavar="X.csv",
    required=True,
    help="Training data")

parser.add_argument(
    "--model-architectures",
    metavar="X.json",
    type=argparse.FileType('r'),
    required=True,
    help="JSON file giving model architectures to assess in cross validation."
    " Can be - to read from stdin")

parser.add_argument(
    "--alleles",
    metavar="ALLELE",
    nargs="+",
    default=None,
    help="Use only the specified alleles")

parser.add_argument(
    "--out-manifest",
    metavar="X.csv",
    help="Write results to the given file")

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
    "--ensemble-size",
    type=int,
    metavar="N",
    required=True,
    help="Number of models to use per allele")

parser.add_argument(
    "--target-tasks",
    type=int,
    metavar="N",
    required=True,
    help="Target number of tasks to submit")

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

parser.add_argument(
    "--dask-scheduler",
    metavar="HOST:PORT",
    help="Host and port of dask distributed scheduler")

parser.add_argument(
    "--parallel-backend",
    choices=("local-threads", "local-processes", "kubeface", "dask"),
    default="local-threads",
    help="Backend to use, default: %(default)s")

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
    if args.parallel_backend == "dask":
        backend = parallelism.DaskDistributedParallelBackend(
            args.dask_scheduler)
    elif args.parallel_backend == "kubeface":
        backend = parallelism.KubefaceParallelBackend(args)
    elif args.parallel_backend == "local-threads":
        backend = parallelism.ConcurrentFuturesParallelBackend(
            args.num_local_threads,
            processes=False)
    elif args.parallel_backend == "local-processes":
        backend = parallelism.ConcurrentFuturesParallelBackend(
            args.num_local_processes,
            processes=True)
    else:
        assert False, args.parallel_backend

    parallelism.set_default_backend(backend)
    print("Using parallel backend: %s" % backend)
    go(args)


def go(args):
    model_architectures = json.loads(args.model_architectures.read())
    logging.info("Read %d model architectures" % len(model_architectures))
    if args.max_models:
        model_architectures = model_architectures[:args.max_models]
        logging.info(
            "Subselected to %d model architectures" % len(model_architectures))

    train_dataset = Dataset.from_csv(args.train_data)
    logging.info("Loaded training data: %s" % train_dataset)

    if args.alleles:
        train_dataset = train_dataset.get_alleles(args.alleles)
        logging.info(
            "Filtered training dataset by allele to: %s" % train_dataset)

    if args.min_samples_per_allele:
        train_dataset = train_dataset.filter_alleles_by_count(
            args.min_samples_per_allele)
        logging.info(
            "Filtered training dataset to alleles with >= %d observations: %s"
            % (args.min_samples_per_allele, train_dataset))

    train_mc = MeasurementCollection.from_dataset(train_dataset)
    model = Class1EnsembleMultiAllelePredictor(
        args.ensemble_size,
        model_architectures)
    model.fit(train_mc, target_tasks=args.target_tasks)
    logging.info("Done fitting.")

    model.write_fit(args.out_manifest, args.out_models_dir)
