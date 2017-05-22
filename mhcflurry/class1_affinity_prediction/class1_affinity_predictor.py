import collections
import time
import hashlib
import json
from os.path import join, exists
from six import string_types

import numpy
import pandas

import mhcnames

from ..encodable_sequences import EncodableSequences
from ..downloads import get_path

from .class1_neural_network import Class1NeuralNetwork


class LazyLoadingClass1NeuralNetwork(object):
    @classmethod
    def wrap(cls, instance):
        if isinstance(instance, cls):
            return instance
        elif isinstance(instance, Class1NeuralNetwork):
            return cls(model=instance)
        raise TypeError("Unsupported type: %s" % instance)

    @classmethod
    def wrap_list(cls, lst):
        return [
            cls.wrap(instance)
            for instance in lst
        ]

    def __init__(self, model=None, config=None, weights_filename=None):
        if model is None:
            assert config is not None
            assert weights_filename is not None
        else:
            assert config is None
            assert weights_filename is None

        self.model = model
        self.config = config
        self.weights_filename = weights_filename

    @property
    def instance(self):
        if self.model is None:
            self.model = Class1NeuralNetwork.from_config(self.config)
            self.model.restore_weights(self.weights_filename)
        return self.model


class Class1AffinityPredictor(object):
    def __init__(
            self,
            allele_to_allele_specific_models=None,
            class1_pan_allele_models=None,
            allele_to_pseudosequence=None,
            manifest_df=None):

        if allele_to_allele_specific_models is None:
            allele_to_allele_specific_models = {}
        if class1_pan_allele_models is None:
            class1_pan_allele_models = []

        if class1_pan_allele_models:
            assert allele_to_pseudosequence, "Pseudosequences required"

        self.allele_to_allele_specific_models = dict(
            (k, LazyLoadingClass1NeuralNetwork.wrap_list(v))
            for (k, v) in allele_to_allele_specific_models.items())
        self.class1_pan_allele_models = (
            LazyLoadingClass1NeuralNetwork.wrap_list(class1_pan_allele_models))
        self.allele_to_pseudosequence = allele_to_pseudosequence

        if manifest_df is None:
            manifest_df = pandas.DataFrame()
            manifest_df["model_name"] = []
            manifest_df["allele"] = []
            manifest_df["config_json"] = []
            manifest_df["model"] = []
        self.manifest_df = manifest_df

    def save(self, models_dir, model_names_to_write=None):
        num_models = len(self.class1_pan_allele_models) + sum(
            len(v) for v in self.allele_to_allele_specific_models.values())
        assert len(self.manifest_df) == num_models, (
            "Manifest seems out of sync with models: %d vs %d entries" % (
                len(self.manifest_df), num_models))

        if model_names_to_write is None:
            # Write all models
            model_names_to_write = self.manifest_df.model_name.values

        sub_manifest_df = self.manifest_df.ix[
            self.manifest_df.model_name.isin(model_names_to_write)
        ]

        for (_, row) in sub_manifest_df.iterrows():
            weights_path = self.weights_path(models_dir, row.model_name)
            row.model.instance.save_weights(weights_path)
            print("Wrote: %s" % weights_path)

        write_manifest_df = self.manifest_df[[
            c for c in self.manifest_df.columns if c != "model"
        ]]
        manifest_path = join(models_dir, "manifest.csv")
        write_manifest_df.to_csv(manifest_path, index=False)
        print("Wrote: %s" % manifest_path)

    @staticmethod
    def model_name(allele, num):
        random_string = hashlib.sha1(
            str(time.time()).encode()).hexdigest()[:16]
        return "%s-%d-%s" % (allele.upper(), num, random_string)

    @staticmethod
    def weights_path(models_dir, model_name):
        return join(
            models_dir,
            "weights_%s.%s" % (
                model_name, Class1NeuralNetwork.weights_filename_extension))

    @staticmethod
    def load(models_dir=None, max_models=None):
        if models_dir is None:
            models_dir = get_path("models_class1", "models")

        manifest_path = join(models_dir, "manifest.csv")
        manifest_df = pandas.read_csv(manifest_path, nrows=max_models)

        allele_to_allele_specific_models = collections.defaultdict(list)
        class1_pan_allele_models = []
        all_models = []
        for (_, row) in manifest_df.iterrows():
            model = LazyLoadingClass1NeuralNetwork(
                config=json.loads(row.config_json),
                weights_filename=Class1AffinityPredictor.weights_path(
                    models_dir, row.model_name)
            )
            if row.allele == "pan-class1":
                class1_pan_allele_models.append(model)
            else:
                allele_to_allele_specific_models[row.allele].append(model)
            all_models.append(model)

        manifest_df["model"] = all_models

        pseudosequences = None
        if exists(join(models_dir, "pseudosequences.csv")):
            pseudosequences = pandas.read_csv(
                join(models_dir, "pseudosequences.csv"),
                index_col="allele").to_dict()

        print(
            "Loaded %d class1 pan allele predictors, %d pseudosequences, and "
            "%d allele specific models: %s" % (
                len(class1_pan_allele_models),
                len(pseudosequences) if pseudosequences else 0,
                sum(len(v) for v in allele_to_allele_specific_models.values()),
                ", ".join(
                    "%s (%d)" % (allele, len(v))
                    for (allele, v)
                    in sorted(allele_to_allele_specific_models.items()))))

        result = Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_pseudosequence=pseudosequences,
            manifest_df=manifest_df)
        return result

    def fit_allele_specific_predictors(
            self,
            n_models,
            architecture_hyperparameters,
            allele,
            peptides,
            affinities,
            models_dir_for_save=None,
            verbose=1):

        allele = mhcnames.normalize_allele_name(allele)
        models = self._fit_predictors(
            n_models=n_models,
            architecture_hyperparameters=architecture_hyperparameters,
            peptides=peptides,
            affinities=affinities,
            allele_pseudosequences=None,
            verbose=verbose)

        if allele not in self.allele_to_allele_specific_models:
            self.allele_to_allele_specific_models[allele] = []

        models_list = []
        for (i, model) in enumerate(models):
            lazy_model = LazyLoadingClass1NeuralNetwork.wrap(model)
            model_name = self.model_name(allele, i)
            models_list.append(model)  # models is a generator
            row = pandas.Series(collections.OrderedDict([
                ("model_name", model_name),
                ("allele", allele),
                ("config_json", json.dumps(model.get_config())),
                ("model", lazy_model),
            ])).to_frame().T
            self.manifest_df = pandas.concat(
                [self.manifest_df, row], ignore_index=True)
            self.allele_to_allele_specific_models[allele].append(lazy_model)
            if models_dir_for_save:
                self.save(
                    models_dir_for_save, model_names_to_write=[model_name])
        return models

    def fit_class1_pan_allele_models(
            self,
            n_models,
            architecture_hyperparameters,
            alleles,
            peptides,
            affinities,
            models_dir_for_save=None,
            verbose=1):

        alleles = pandas.Series(alleles).map(mhcnames.normalize_allele_name)
        allele_pseudosequences = alleles.map(self.allele_to_pseudosequence)

        models = self._fit_predictors(
            n_models=n_models,
            architecture_hyperparameters=architecture_hyperparameters,
            peptides=peptides,
            affinities=affinities,
            allele_pseudosequences=allele_pseudosequences,
            verbose=verbose)

        for (i, model) in enumerate(models):
            lazy_model = LazyLoadingClass1NeuralNetwork.wrap(model)
            model_name = self.model_name("pan-class1", i)
            self.class1_pan_allele_models.append(lazy_model)
            row = pandas.Series(collections.OrderedDict([
                ("model_name", model_name),
                ("allele", "pan-class1"),
                ("config_json", json.dumps(model.get_config())),
                ("model", lazy_model),
            ])).to_frame().T
            self.manifest_df = pandas.concat(
                [self.manifest_df, row], ignore_index=True)
            if models_dir_for_save:
                self.save(
                    models_dir_for_save, model_names_to_write=[model_name])
        return models

    def _fit_predictors(
            self,
            n_models,
            architecture_hyperparameters,
            peptides,
            affinities,
            allele_pseudosequences,
            verbose=1):

        encodable_peptides = EncodableSequences.create(peptides)
        for i in range(n_models):
            print("Training model %d / %d" % (i + 1, n_models))
            model = Class1NeuralNetwork(**architecture_hyperparameters)
            model.fit(
                encodable_peptides,
                affinities,
                allele_pseudosequences=allele_pseudosequences,
                verbose=verbose)
            yield model

    def predict(self, peptides, alleles=None, allele=None):
        df = self.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            allele=allele
        )
        return df.prediction.values

    def predict_to_dataframe(
            self,
            peptides,
            alleles=None,
            allele=None,
            include_individual_model_predictions=False):
        if isinstance(peptides, string_types):
            raise TypeError("peptides must be a list or array, not a string")
        if isinstance(alleles, string_types):
            raise TypeError("alleles must be a list or array, not a string")
        if allele is not None:
            if alleles is not None:
                raise ValueError("Specify exactly one of allele or alleles")
            alleles = [allele] * len(peptides)

        df = pandas.DataFrame({
            'peptide': peptides,
            'allele': alleles,
        })
        df["normalized_allele"] = df.allele.map(
            mhcnames.normalize_allele_name)

        if self.class1_pan_allele_models:
            allele_pseudosequences = df.normalized_allele.map(
                self.allele_to_pseudosequence)
            encodable_peptides = EncodableSequences.create(
                df.peptide.values)
            for (i, model) in enumerate(self.class1_pan_allele_models):
                df["model_pan_%d" % i] = model.instance.predict(
                    encodable_peptides,
                    allele_pseudosequences=allele_pseudosequences)

        if self.allele_to_allele_specific_models:
            for allele in df.normalized_allele.unique():
                mask = (df.normalized_allele == allele).values
                allele_peptides = EncodableSequences.create(
                    df.ix[mask].peptide.values)
                models = self.allele_to_allele_specific_models.get(allele, [])
                for (i, model) in enumerate(models):
                    df.loc[
                        mask, "model_single_%d" % i
                    ] = model.instance.predict(allele_peptides)

        # Geometric mean
        df_predictions = df[
            [c for c in df.columns if c.startswith("model_")]
        ]
        logs = numpy.log(df_predictions)
        log_means = logs.mean(1)
        df["prediction"] = numpy.exp(log_means)
        df["prediction_low"] = numpy.exp(logs.quantile(0.05, axis=1))
        df["prediction_high"] = numpy.exp(logs.quantile(0.95, axis=1))

        del df["normalized_allele"]
        if include_individual_model_predictions:
            columns = sorted(df.columns, key=lambda c: c.startswith('model_'))
        else:
            columns = [
                c for c in df.columns if c not in df_predictions.columns
            ]
        return df[columns]