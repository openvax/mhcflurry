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

"""
Ensemble of allele specific MHC Class I binding affinity predictors
"""
from __future__ import (
    print_function,
    division,
    absolute_import,
)

import pickle
import os
import math
import logging
import collections
import time
import numpy
from functools import partial

import pandas

from ..hyperparameters import HyperparameterDefaults
from ..class1_allele_specific import Class1BindingPredictor, scoring
from ..downloads import get_path
from .. import parallelism, common

MEASUREMENT_COLLECTION_HYPERPARAMETER_DEFAULTS = HyperparameterDefaults(
    include_ms=True,
    ms_hit_affinity=1.0,
    ms_decoy_affinity=20000.0)

IMPUTE_HYPERPARAMETER_DEFAULTS = HyperparameterDefaults(
    impute_method='mice',
    impute_min_observations_per_peptide=5,
    impute_min_observations_per_allele=5,
    imputer_args={"n_burn_in": 5, "n_imputations": 25})

HYPERPARAMETER_DEFAULTS = (
    HyperparameterDefaults(
        impute=True,
        architecture_num=None)
    .extend(MEASUREMENT_COLLECTION_HYPERPARAMETER_DEFAULTS)
    .extend(IMPUTE_HYPERPARAMETER_DEFAULTS)
    .extend(Class1BindingPredictor.hyperparameter_defaults))


CACHED_PREDICTOR = None
CACHED_PREDICTOR_PATH = None


def supported_alleles():
    """
    Return a list of the names of the alleles for which there are trained
    predictors in the default laoder.
    """
    return get_downloaded_predictor().supported_alleles


def get_downloaded_predictor():
    """
    Return a Class1AlleleSpecificPredictorLoader that uses downloaded models.
    """
    global CACHED_PREDICTOR, CACHED_PREDICTOR_PATH

    # Some of the unit tests manipulate the downloads directory configuration
    # so get_path here may return different results in the same Python process.
    # For this reason we check the path and invalidate the loader if it's
    # different.
    path = get_path("models_class1_allele_specific_ensemble")
    if CACHED_PREDICTOR_PATH != path:
        CACHED_PREDICTOR_PATH = path
        CACHED_PREDICTOR = (
            Class1EnsembleMultiAllelePredictor
                .load_from_download_directory(path))
    return CACHED_PREDICTOR


def call_fit_and_test(args):
    return fit_and_test(*args)


def fit_and_test(
        parallel_backend,
        fold_num,
        train_mc_remote_object,
        imputed_mc_remote_object,
        test_mc_remote_object,
        allele_and_hyperparameter_pairs):

    logging.info(
        "Fit and test: fold=%d train=%s,%s test=%s alleles/models [%d]=%s" % (
            fold_num,
            train_mc_remote_object.value,
            imputed_mc_remote_object,
            test_mc_remote_object.value,
            len(allele_and_hyperparameter_pairs),
            "\n".join("Allele: %s, hyperparameters: %s" % (
                allele, hyperparameters)
                for (allele, hyperparameters)
                in allele_and_hyperparameter_pairs)))

    assert len(train_mc_remote_object.value.df) > 0
    assert len(test_mc_remote_object.value.df) > 0

    train_mc_hash = common.dataframe_cryptographic_hash(
        train_mc_remote_object.value.df)

    imputed_mc_hash = None
    if imputed_mc_remote_object is not None:
        imputed_mc_hash = common.dataframe_cryptographic_hash(
            imputed_mc_remote_object.value.df)
    test_mc_hash = common.dataframe_cryptographic_hash(
        test_mc_remote_object.value.df)

    common_result_entries = {
        'fold_num': fold_num,
        'all_alleles_train_data_hash': train_mc_hash,
        'all_alleles_imputed_data_hash': imputed_mc_hash,
        'all_alleles_test_data_hash': test_mc_hash,
    }

    results = []
    for (i, (allele, all_hyperparameters)) in enumerate(
            allele_and_hyperparameter_pairs):
        logging.info("Model %d / %d: allele=%s hyperparameters=%s" % (
            i + 1,
            len(allele_and_hyperparameter_pairs),
            allele,
            str(all_hyperparameters)))

        start = time.time()
        measurement_collection_hyperparameters = (
            MEASUREMENT_COLLECTION_HYPERPARAMETER_DEFAULTS.subselect(
                all_hyperparameters))
        model_hyperparameters = (
            Class1BindingPredictor.hyperparameter_defaults.subselect(
                all_hyperparameters))

        train_dataset = (
            train_mc_remote_object
            .value
            .select_allele(allele)
            .to_dataset(**measurement_collection_hyperparameters))
        if all_hyperparameters['impute'] and (
                allele in
                imputed_mc_remote_object.value.alleles):
            imputed_train_dataset = (
                imputed_mc_remote_object
                .value
                .select_allele(allele)
                .to_dataset(**measurement_collection_hyperparameters))
        else:
            imputed_train_dataset = None
        test_dataset = (
            test_mc_remote_object
            .value
            .select_allele(allele)
            .to_dataset(**measurement_collection_hyperparameters))

        assert len(train_dataset) > 0
        assert len(test_dataset) > 0

        model = Class1BindingPredictor(**model_hyperparameters)

        train_start = time.time()
        model.fit_dataset(
            train_dataset,
            pretraining_dataset=imputed_train_dataset)
        train_end = time.time()
        predictions = model.predict(test_dataset.peptides)
        test_end = time.time()
        scores = scoring.make_scores(
            test_dataset.affinities, predictions)

        result = dict(common_result_entries)
        result.update({
            'allele': allele,
            'hyperparameters': all_hyperparameters,
            'model': parallel_backend.remote_object(model),
            'scores': scores,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset),
            'pretrain_size': (
                0 if imputed_train_dataset is None
                else len(imputed_train_dataset)),
            'train_time': train_end - train_start,
            'predict_time': test_end - train_end,
        })
        results.append(result)
        logging.info("Done training model in %0.2f sec" % (
            time.time() - start))
    return results


def impute(parallel_backend, hyperparameters, measurement_collection):
    return parallel_backend.remote_object(
        measurement_collection.impute(**hyperparameters))


class Class1EnsembleMultiAllelePredictor(object):
    @staticmethod
    def load_from_download_directory(directory):
        return Class1EnsembleMultiAllelePredictor.load_fit(
            os.path.join(directory, "models"),
            os.path.join(directory, "selected_models.csv"),
        )

    @staticmethod
    def load_fit(path_to_models_dir, path_to_manifest):
        manifest_df = pandas.read_csv(path_to_manifest, index_col="model_name")
        # Convert string-serialized dicts into Python objects.
        manifest_df["hyperparameters"] = [
            eval(s) for s in manifest_df.hyperparameters
        ]
        hyperparameters_to_search = list(dict(
            (row.hyperparameters_architecture_num, row.hyperparameters)
            for (_, row) in manifest_df.iterrows()
        ).values())
        (ensemble_size,) = list(manifest_df.ensemble_size.unique())
        assert (
            manifest_df.ix[manifest_df.weight > 0]
            .groupby("allele")
            .weight
            .count() == ensemble_size).all()
        result = Class1EnsembleMultiAllelePredictor(
            ensemble_size=ensemble_size,
            hyperparameters_to_search=hyperparameters_to_search)
        result.manifest_df = manifest_df
        result.allele_to_models = {}
        result.models_dir = os.path.abspath(path_to_models_dir)
        return result

    def __init__(self, ensemble_size, hyperparameters_to_search):
        self.imputation_hyperparameters = None  # None indicates no imputation
        self.hyperparameters_to_search = []
        for (num, params) in enumerate(hyperparameters_to_search):
            params = dict(params)
            params["architecture_num"] = num
            params = HYPERPARAMETER_DEFAULTS.with_defaults(params)
            self.hyperparameters_to_search.append(params)

            if params['impute']:
                imputation_args = IMPUTE_HYPERPARAMETER_DEFAULTS.subselect(
                    params)
                if self.imputation_hyperparameters is None:
                    self.imputation_hyperparameters = imputation_args
                if self.imputation_hyperparameters != imputation_args:
                    raise NotImplementedError(
                        "Only one set of imputation parameters is supported: "
                        "%s != %s" % (
                            str(self.imputation_hyperparameters),
                            str(imputation_args)))

        self.ensemble_size = ensemble_size
        self.manifest_df = None
        self.allele_to_models = None
        self.models_dir = None

    @property
    def supported_alleles(self):
        return list(
            self.manifest_df.ix[self.manifest_df.weight > 0].allele.unique())

    def description(self):
        lines = []
        kvs = []

        def kv(key, value):
            kvs.append((key, value))

        kv("ensemble size", self.ensemble_size)
        kv("num architectures considered",
            len(self.hyperparameters_to_search))
        if self.allele_to_models is not None:
            kv("supported alleles", " ".join(self.supported_alleles))
        kv("models dir", self.models_dir)

        lines.append("%s Ensemble: %s" % (
            "Untrained" if self.allele_to_models is None else "Trained",
            self))
        for (key, value) in kvs:
            lines.append("* %s: %s" % (key, value))

        if self.manifest_df is not None:
            models_used = self.manifest_df.ix[self.manifest_df.weight > 0]

            ignored_properties = set(['hyperparameters', 'scores'])
            lines.append("* Attributes common to all models:")
            unique = None
            for col in models_used.columns:
                unique = models_used[col].map(str).unique()
                if len(unique) == 1:
                    lines.append("\t%s: %s" % (col, unique[0]))
                    ignored_properties.add(col)
            if unique is None:
                lines.append("\t(none)")

            for (allele, manifest_rows) in models_used.groupby("allele"):
                lines.append("***")
                for (i, (name, row)) in enumerate(manifest_rows.iterrows()):
                    lines.append("* %s model %d: %s" % (
                        allele, i + 1, name))
                    for (k, v) in row.iteritems():
                        if k not in ignored_properties:
                            lines.append("\t%s: %s" % (k, v))
                    lines.append("")
        return "\n".join(lines)

    def models_for_allele(self, allele):
        if allele not in self.allele_to_models:
            model_names = self.manifest_df.ix[
                (self.manifest_df.weight > 0) &
                (self.manifest_df.allele == allele)
            ].index
            if len(model_names) == 0:
                raise ValueError(
                    "Unsupported allele: %s. Supported alleles: %s" % (
                        allele,
                        ", ".join(self.supported_alleles)))
            assert len(model_names) == self.ensemble_size
            models = []
            for name in model_names:
                filename = os.path.join(
                    self.models_dir, "%s.pickle" % name)
                with open(filename, 'rb') as fd:
                    model = pickle.load(fd)
                    assert model.name == name
                    models.append(model)
            self.allele_to_models[allele] = models
        result = self.allele_to_models[allele]
        assert len(result) == self.ensemble_size
        return result

    def write_fit(
            self,
            models_dir=None,
            selected_models_csv=None,
            all_models_csv=None):
        if all_models_csv:
            self.manifest_df.to_csv(all_models_csv)
            logging.debug("Wrote: %s" % all_models_csv)
        if selected_models_csv:
            self.manifest_df.ix[
                self.manifest_df.weight > 0
            ].to_csv(selected_models_csv)
            logging.debug("Wrote: %s" % selected_models_csv)

        if models_dir:
            models_written = []
            for (allele, models) in self.allele_to_models.items():
                for model in models:
                    filename = os.path.join(
                        models_dir, "%s.pickle" % model.name)
                    with open(filename, 'wb') as fd:
                        pickle.dump(model, fd)
                    logging.debug("Wrote: %s" % filename)
                    models_written.append(model.name)
            assert set(models_written) == set(
                self.manifest_df.ix[self.manifest_df.weight > 0].index)

    def predict(self, measurement_collection):
        result = pandas.Series(
            index=measurement_collection.df.index)
        for (allele, sub_df) in measurement_collection.df.groupby("allele"):
            result.loc[sub_df.index] = self.predict_for_allele(
                allele, sub_df.peptide.values)
        assert not result.isnull().any()
        return result

    def predict_for_allele(self, allele, peptides):
        values = [
            model.predict(peptides)
            for model in self.models_for_allele(allele)
        ]

        # Geometric mean
        result = numpy.exp(numpy.nanmean(numpy.log(values), axis=0))
        assert len(result) == len(peptides)
        return result

    def fit(
            self,
            measurement_collection,
            parallel_backend=None,
            target_tasks=1):
        if parallel_backend is None:
            parallel_backend = parallelism.get_default_backend()

        fit_name = time.asctime().replace(" ", "_")
        assert len(measurement_collection.df) > 0

        splits = measurement_collection.half_splits(
            self.ensemble_size, random_state=0)

        if self.imputation_hyperparameters is not None:
            logging.info("Imputing: %d tasks, imputation args: %s" % (
                len(splits), str(self.imputation_hyperparameters)))
            imputed_trains = list(parallel_backend.map(
                partial(
                    impute, parallel_backend, self.imputation_hyperparameters),
                [train for (train, test) in splits]))
            logging.info("Imputation completed.")
        else:
            logging.info("No imputation required.")
            imputed_trains = None

        assert len(splits) == self.ensemble_size, len(splits)

        alleles = set(measurement_collection.df.allele.unique())

        total_work = (
            len(alleles) *
            self.ensemble_size *
            len(self.hyperparameters_to_search))
        work_per_task = int(math.ceil(total_work / target_tasks))
        tasks = []
        for (fold_num, (train_split, test_split)) in enumerate(splits):
            assert len(train_split.df) > 0
            assert len(test_split.df) > 0
            train_remote_object = parallel_backend.remote_object(train_split)
            test_remote_object = parallel_backend.remote_object(test_split)
            imputed_train_remote_object = None
            if imputed_trains is not None:
                imputed_train_remote_object = imputed_trains[fold_num]

            task_allele_model_pairs = []

            def make_task():
                if task_allele_model_pairs:
                    tasks.append((
                        parallel_backend,
                        fold_num,
                        train_remote_object,
                        imputed_train_remote_object,
                        test_remote_object,
                        list(task_allele_model_pairs)))
                    task_allele_model_pairs[:] = []

            assert all(
                allele in set(train_split.df.allele.unique())
                for allele in alleles), (
                "%s not in %s" % (
                    alleles, set(train_split.df.allele.unique())))
            assert all(
                allele in set(test_split.df.allele.unique())
                for allele in alleles), (
                "%s not in %s" % (
                    alleles, set(test_split.df.allele.unique())))

            for model in self.hyperparameters_to_search:
                for allele in alleles:
                    task_allele_model_pairs.append((allele, model))
                    if len(task_allele_model_pairs) > work_per_task:
                        make_task()
            make_task()
            assert not task_allele_model_pairs

        allele_models_per_task = numpy.array([
            len(task[-1]) for task in tasks
        ])
        logging.info(
            "Training and scoring models: %d tasks (target was %d), "
            "total work: %d alleles * %d ensemble size * %d models = %d, "
            "allele/models per task: (min=%d mean=%f max=%d)" % (
                len(tasks),
                target_tasks,
                len(alleles),
                self.ensemble_size,
                len(self.hyperparameters_to_search),
                total_work,
                allele_models_per_task.min(),
                allele_models_per_task.max(),
                allele_models_per_task.mean()))

        assert len(tasks) > 0
        results = parallel_backend.map(call_fit_and_test, tasks)

        # fold number -> allele -> best model
        results_per_fold = [
            {}
            for _ in range(len(splits))
        ]
        next_model_num = 1
        manifest_rows = []
        for result in results:
            logging.debug("Received task result with %d items." % len(result))
            for item in result:
                item['model_name'] = "%s.%d.%s" % (
                    item['allele'], next_model_num, fit_name)
                next_model_num += 1

                scores = pandas.Series(item['scores'])
                item['summary_score'] = scores.fillna(0).sum()
                fold_results = results_per_fold[item['fold_num']]
                allele = item['allele']
                current_best = float('-inf')
                if allele in fold_results:
                    current_best = fold_results[allele]['summary_score']

                if item['summary_score'] > current_best:
                    logging.info("Updating current best: %s" % str(item))
                    fold_results[allele] = item

                manifest_entry = dict(item)
                del manifest_entry['model']
                for key in ['hyperparameters', 'scores']:
                    for (sub_key, value) in item[key].items():
                        manifest_entry["%s_%s" % (key, sub_key)] = value
                manifest_rows.append(manifest_entry)

        assert len(manifest_rows) > 0
        manifest_df = pandas.DataFrame(manifest_rows)
        manifest_df.index = manifest_df.model_name
        del manifest_df["model_name"]
        manifest_df["weight"] = 0.0
        manifest_df["ensemble_size"] = self.ensemble_size

        logging.info("Done collecting results.")

        self.allele_to_models = collections.defaultdict(list)
        for fold_results in results_per_fold:
            assert set(fold_results) == set(alleles), (
                "%s != %s" % (set(fold_results), set(alleles)))
            for (allele, item) in fold_results.items():
                model = item['model'].value
                model.name = item['model_name']
                self.allele_to_models[allele].append(model)
                manifest_df.loc[model.name, "weight"] = 1.0

        self.manifest_df = manifest_df
