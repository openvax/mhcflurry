from __future__ import print_function

from os.path import join, exists, abspath
from os import mkdir
from socket import gethostname
from getpass import getuser

import time
import json
import hashlib
import logging
import collections
from six import string_types

import numpy
import pandas

from .version import __version__
from .class1_neural_network import DEFAULT_PREDICT_BATCH_SIZE
from .flanking_encoding import FlankingEncoding
from .downloads import get_default_class1_cleavage_models_dir
from .class1_cleavage_neural_network import Class1CleavageNeuralNetwork
from .common import save_weights, load_weights, NumpyJSONEncoder


class Class1CleavagePredictor(object):
    def __init__(
            self,
            models,
            manifest_df=None,
            metadata_dataframes=None):
        self.models = models
        self._manifest_df = manifest_df
        self.metadata_dataframes = (
            dict(metadata_dataframes) if metadata_dataframes else {})

    def add_models(self, models):
        new_model_names = []
        original_manifest = self.manifest_df
        new_manifest_rows = []
        for model in models:
            model_name = self.model_name(len(self.models))
            row = pandas.Series(collections.OrderedDict([
                ("model_name", model_name),
                ("config_json", json.dumps(
                    model.get_config(), cls=NumpyJSONEncoder)),
                ("model", model),
            ])).to_frame().T
            new_manifest_rows.append(row)
            self.models.append(model)
            new_model_names.append(model_name)

        self._manifest_df = pandas.concat(
            [original_manifest] + new_manifest_rows,
            ignore_index=True)

        self.check_consistency()
        return new_model_names


    @property
    def manifest_df(self):
        """
        A pandas.DataFrame describing the models included in this predictor.

        Returns
        -------
        pandas.DataFrame
        """
        if self._manifest_df is None:
            rows = []
            for (i, model) in enumerate(self.models):
                model_config = model.get_config()
                rows.append((
                    self.model_name(i),
                    json.dumps(model_config, cls=NumpyJSONEncoder),
                    model
                ))
            self._manifest_df = pandas.DataFrame(
                rows,
                columns=["model_name", "config_json", "model"])
        return self._manifest_df

    @staticmethod
    def model_name(num):
        """
        Generate a model name

        Returns
        -------
        string

        """
        random_string = hashlib.sha1(
            str(time.time()).encode()).hexdigest()[:16]
        return "CLEAVAGE-CLASSI-%d-%s" % (
            num,
            random_string)

    @staticmethod
    def weights_path(models_dir, model_name):
        """
        Generate the path to the weights file for a model

        Parameters
        ----------
        models_dir : string
        model_name : string

        Returns
        -------
        string
        """
        return join(models_dir, "weights_%s.npz" % model_name)

    def predict(
            self,
            peptides,
            n_flanks,
            c_flanks,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):

        return self.predict_to_dataframe(
            peptides=peptides,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            batch_size=batch_size).score.values

    def predict_to_dataframe(
            self,
            peptides,
            n_flanks,
            c_flanks,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):

        sequences = FlankingEncoding(
            peptides=peptides, n_flanks=n_flanks, c_flanks=c_flanks)
        return self.predict_to_dataframe_encoded(
            sequences=sequences, batch_size=batch_size)

    def predict_to_dataframe_encoded(
            self, sequences, batch_size=DEFAULT_PREDICT_BATCH_SIZE):

        score_array = []

        for (i, network) in enumerate(self.models):
            predictions = network.predict_encoded(
                encoded, batch_size=batch_size)
            score_array.append(predictions)

        score_array = numpy.array(score_array)

        result_df = pandas.DataFrame({
            "peptide": encoded.dataframe.peptide,
            "n_flank": encoded.dataframe.n_flank,
            "c_flank": encoded.dataframe.c_flank,
            "score": numpy.mean(score_array, axis=0),
        })
        return result_df

    def check_consistency(self):
        """
        Verify that self.manifest_df is consistent with instance variables.

        Currently only checks for agreement on the total number of models.

        Throws AssertionError if inconsistent.
        """
        assert len(self.manifest_df) == len(self.models), (
            "Manifest seems out of sync with models: %d vs %d entries: \n%s"% (
                len(self.manifest_df),
                len(self.models),
                str(self.manifest_df)))

    def save(self, models_dir, model_names_to_write=None, write_metadata=True):
        """
        Serialize the predictor to a directory on disk. If the directory does
        not exist it will be created.

        The serialization format consists of a file called "manifest.csv" with
        the configurations of each Class1CleavageNeuralNetwork, along with
        per-network files giving the model weights.

        Parameters
        ----------
        models_dir : string
            Path to directory. It will be created if it doesn't exist.
        """
        self.check_consistency()

        if model_names_to_write is None:
            # Write all models
            model_names_to_write = self.manifest_df.model_name.values

        if not exists(models_dir):
            mkdir(models_dir)

        sub_manifest_df = self.manifest_df.loc[
            self.manifest_df.model_name.isin(model_names_to_write)
        ].copy()

        # Network JSON configs may have changed since the models were added,
        # so we update the JSON configs here also.
        updated_network_config_jsons = []
        for (_, row) in sub_manifest_df.iterrows():
            updated_network_config_jsons.append(
                json.dumps(row.model.get_config(), cls=NumpyJSONEncoder))
            weights_path = self.weights_path(models_dir, row.model_name)
            save_weights(row.model.get_weights(), weights_path)
            logging.info("Wrote: %s", weights_path)
        sub_manifest_df["config_json"] = updated_network_config_jsons
        self.manifest_df.loc[
            sub_manifest_df.index,
            "config_json"
        ] = updated_network_config_jsons

        write_manifest_df = self.manifest_df[[
            c for c in self.manifest_df.columns if c != "model"
        ]]
        manifest_path = join(models_dir, "manifest.csv")
        write_manifest_df.to_csv(manifest_path, index=False)
        logging.info("Wrote: %s", manifest_path)

        if write_metadata:
            # Write "info.txt"
            info_path = join(models_dir, "info.txt")
            rows = [
                ("trained on", time.asctime()),
                ("package   ", "mhcflurry %s" % __version__),
                ("hostname  ", gethostname()),
                ("user      ", getuser()),
            ]
            pandas.DataFrame(rows).to_csv(
                info_path, sep="\t", header=False, index=False)

            if self.metadata_dataframes:
                for (name, df) in self.metadata_dataframes.items():
                    metadata_df_path = join(models_dir, "%s.csv.bz2" % name)
                    df.to_csv(metadata_df_path, index=False, compression="bz2")

    @classmethod
    def load(cls, models_dir=None, max_models=None):
        """
        Deserialize a predictor from a directory on disk.

        Parameters
        ----------
        models_dir : string
            Path to directory. If unspecified the default downloaded models are
            used.

        max_models : int, optional
            Maximum number of models to load

        Returns
        -------
        `Class1CleavagePredictor` instance
        """
        if models_dir is None:
            models_dir = get_default_class1_cleavage_models_dir()

        manifest_path = join(models_dir, "manifest.csv")
        manifest_df = pandas.read_csv(manifest_path, nrows=max_models)

        models = []
        for (_, row) in manifest_df.iterrows():
            weights_filename = cls.weights_path(models_dir, row.model_name)
            config = json.loads(row.config_json)
            model = Class1CleavageNeuralNetwork.from_config(
                config,
                weights=load_weights(abspath(weights_filename)))
            models.append(model)

        manifest_df["model"] = models

        logging.info("Loaded %d class1 cleavage models", len(models))
        result = cls(
            models=models,
            manifest_df=manifest_df)
        return result
