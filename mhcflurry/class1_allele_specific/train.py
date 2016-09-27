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

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import collections
import logging
import time
import socket
import math

import numpy
import pandas

import mhcflurry

from .scoring import make_scores
from .class1_binding_predictor import Class1BindingPredictor
from ..hyperparameters import HyperparameterDefaults
from ..parallelism import get_default_backend


TRAIN_HYPERPARAMETER_DEFAULTS = HyperparameterDefaults(impute=False)
HYPERPARAMETER_DEFAULTS = (
    Class1BindingPredictor.hyperparameter_defaults
    .extend(TRAIN_HYPERPARAMETER_DEFAULTS))


AlleleSpecificTrainTestFold = collections.namedtuple(
    "AlleleSpecificTrainTestFold",
    "allele train imputed_train test")


def impute_and_select_allele(dataset, imputer, allele=None, **kwargs):
    '''
    Run imputation and optionally filter to the specified allele.

    Useful as a parallelized task where we want to filter to the desired
    data *before* sending the result back to the master process.

    Parameters
    -----------
    dataset : mhcflurry.Dataset

    imputer : object or string
        See Dataset.impute_missing_values

    allele : string [optional]
        Allele name to subselect to after imputation

    **kwargs : passed on to dataset.impute_missing_values

    Returns
    -----------
    list of dict
    '''
    result = dataset.impute_missing_values(imputer, **kwargs)

    if allele is not None:
        try:
            result = result.get_allele(allele)
        except KeyError:
            result = None
    return result


def train_and_test_one_model(model_description, folds, **kwargs):
    '''
    Train one model on some number of folds.

    Parameters
    -----------
    model_description : dict of model hyperparameters

    folds : list of AlleleSpecificTrainTestFold

    **kwargs : passed on to train_and_test_one_model_one_fold

    Returns
    -----------
    list of dict giving the train and test results for each fold
    '''
    logging.info("Training 1 model on %d folds: %s" % (len(folds), folds))

    return [
        train_and_test_one_model_one_fold(
            model_description,
            fold.train,
            fold.test,
            fold.imputed_train,
            **kwargs)
        for fold in folds
    ]


def train_and_test_one_model_one_fold(
        model_description,
        train_dataset,
        test_dataset=None,
        imputed_train_dataset=None,
        return_train_scores=True,
        return_predictor=False,
        return_train_predictions=False,
        return_test_predictions=False):
    '''
    Task for instantiating, training, and testing one model on one fold.

    Parameters
    -----------
    model_description : dict of model parameters

    train_dataset : mhcflurry.Dataset
        Dataset to train on. Must include only one allele.

    test_dataset : mhcflurry.Dataset, optional
        Dataset to test on. Must include only one allele. If not specified
        no testing is performed.

    imputed_train_dataset : mhcflurry.Dataset, optional
        Required only if model_description["impute"] == True

    return_train_scores : boolean
        Calculate and include in the result dict the auc/f1/tau scores on the
        training data.

    return_predictor : boolean
        Calculate and include in the result dict the trained predictor.

    return_train_predictions : boolean
        Calculate and include in the result dict the model predictions on the
        train data.

    return_test_predictions : boolean
        Calculate and include in the result dict the model predictions on the
        test data.

    Returns
    -----------
    dict
    '''
    assert len(train_dataset.unique_alleles()) == 1, "Multiple train alleles"
    allele = train_dataset.alleles[0]
    if test_dataset is not None:
        assert len(train_dataset.unique_alleles()) == 1, \
            "Multiple test alleles"
        assert train_dataset.alleles[0] == allele, \
            "Wrong test allele %s != %s" % (train_dataset.alleles[0], allele)
    if imputed_train_dataset is not None:
        assert len(imputed_train_dataset.unique_alleles()) == 1, \
            "Multiple imputed train alleles"
        assert imputed_train_dataset.alleles[0] == allele, \
            "Wrong imputed train allele %s != %s" % (
                imputed_train_dataset.alleles[0], allele)

    if model_description["impute"]:
        assert imputed_train_dataset is not None

    # Make a predictor
    model_params = dict(model_description)
    fraction_negative = model_params.pop("fraction_negative")
    impute = model_params.pop("impute")
    n_training_epochs = model_params.pop("n_training_epochs")
    pretrain_decay = model_params.pop("pretrain_decay")
    batch_size = model_params.pop("batch_size")
    max_ic50 = model_params.pop("max_ic50")

    logging.info(
        "%10s train_size=%d test_size=%d impute=%s model=%s" %
        (allele,
            len(train_dataset),
            len(test_dataset) if test_dataset is not None else 0,
            impute,
            model_description))

    predictor = mhcflurry.Class1BindingPredictor(
        max_ic50=max_ic50,
        **model_params)

    # Train predictor
    fit_time = -time.time()
    predictor.fit_dataset(
        train_dataset,
        pretrain_decay=lambda epoch: eval(pretrain_decay, {
            'epoch': epoch, 'numpy': numpy}),
        pretraining_dataset=imputed_train_dataset if impute else None,
        verbose=True,
        batch_size=batch_size,
        n_training_epochs=n_training_epochs,
        n_random_negative_samples=int(fraction_negative * len(train_dataset)))
    fit_time += time.time()

    result = {
        'fit_time': fit_time,
        'fit_host': socket.gethostname(),
    }

    if return_predictor:
        result['predictor'] = predictor

    if return_train_scores or return_train_predictions:
        train_predictions = predictor.predict(train_dataset.peptides)
        if return_train_scores:
            result['train_scores'] = make_scores(
                train_dataset.affinities,
                train_predictions,
                max_ic50=model_description["max_ic50"])
        if return_train_predictions:
            result['train_predictions'] = train_predictions

    if test_dataset is not None:
        test_predictions = predictor.predict(test_dataset.peptides)
        result['test_scores'] = make_scores(
            test_dataset.affinities,
            test_predictions,
            max_ic50=model_description["max_ic50"])
        if return_test_predictions:
            result['test_predictions'] = test_predictions
    logging.info("Training result: %s" % result)
    return result


def train_across_models_and_folds(
        folds,
        model_descriptions,
        cartesian_product_of_folds_and_models=True,
        return_predictors=False,
        folds_per_task=1,
        parallel_backend=None):
    '''
    Train and optionally test any number of models across any number of folds.

    Parameters
    -----------
    folds : list of AlleleSpecificTrainTestFold

    model_descriptions : list of dict
        Models to test

    cartesian_product_of_folds_and_models : boolean, optional
        If true, then a predictor is treained for each fold and model
        description.
        If false, then len(folds) must equal len(model_descriptions), and
        the i'th model is trained on the i'th fold.

    return_predictors : boolean, optional
        Include the trained predictors in the result.

    parallel_backend : mhcflurry.parallelism.ParallelBackend, optional
        Futures implementation to use for running on multiple threads,
        processes, or nodes

    Returns
    -----------
    pandas.DataFrame
    '''
    if parallel_backend is None:
        parallel_backend = get_default_backend()

    if cartesian_product_of_folds_and_models:
        tasks_per_model = int(math.ceil(float(len(folds)) / folds_per_task))
        fold_index_groups = [[] for _ in range(tasks_per_model)]
        index_group = 0
        for index in range(len(folds)):
            fold_index_groups[index_group].append(index)
            index_group += 1
            if index_group == len(fold_index_groups):
                index_group = 0

        task_model_and_fold_indices = [
            (model_num, group)
            for group in fold_index_groups
            for model_num in range(len(model_descriptions))
        ]
    else:
        assert len(folds) == len(model_descriptions), \
            "folds and models have different lengths and " \
            "cartesian_product_of_folds_and_models is False"

        task_model_and_fold_indices = [
            (num, [num])
            for num in range(len(folds))
        ]

    logging.info("Training %d architectures on %d folds = %d tasks." % (
        len(model_descriptions), len(folds), len(task_model_and_fold_indices)))

    def train_and_test_one_model_task(model_and_fold_nums_pair):
        (model_num, fold_nums) = model_and_fold_nums_pair
        return train_and_test_one_model(
            model_descriptions[model_num],
            [folds[i] for i in fold_nums],
            return_predictor=return_predictors)

    task_results = parallel_backend.map(
        train_and_test_one_model_task,
        task_model_and_fold_indices)

    logging.info("Done.")

    results_dict = collections.OrderedDict()

    def column(key, value):
        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(value)

    for ((model_num, fold_nums), task_results_for_folds) in zip(
            task_model_and_fold_indices, task_results):
        for (fold_num, task_result) in zip(fold_nums, task_results_for_folds):
            fold = folds[fold_num]
            model_description = model_descriptions[model_num]

            column("allele", fold.allele)
            column("fold_num", fold_num)
            column("model_num", model_num)

            column("train_size", len(fold.train))

            column(
                "test_size",
                len(fold.test) if fold.test is not None else None)

            column(
                "imputed_train_size",
                len(fold.imputed_train)
                if fold.imputed_train is not None else None)

            # Scores
            for score_kind in ['train', 'test']:
                field = "%s_scores" % score_kind
                for (score, value) in task_result.pop(field, {}).items():
                    column("%s_%s" % (score_kind, score), value)

            # Misc. fields
            for (key, value) in task_result.items():
                column(key, value)

            # Model parameters
            for (model_param, value) in model_description.items():
                column("model_%s" % model_param, value)

    results_df = pandas.DataFrame(results_dict)
    return results_df
