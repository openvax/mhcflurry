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

import numpy
import pandas

from .version import __version__
from .class1_neural_network import DEFAULT_PREDICT_BATCH_SIZE
from .flanking_encoding import FlankingEncoding
from .downloads import get_default_class1_processing_models_dir
from .class1_processing_neural_network import Class1ProcessingNeuralNetwork
from .common import save_weights, load_weights, NumpyJSONEncoder


class Class1ProcessingPredictor(object):
    """
    User-facing interface to antigen processing prediction.

    Delegates to an ensemble of Class1ProcessingNeuralNetwork instances.
    """
    def __init__(
            self,
            models,
            manifest_df=None,
            metadata_dataframes=None,
            provenance_string=None):
        """
        Instantiate a new Class1ProcessingPredictor

        Users will generally call load() to restore a saved predictor rather
        than using this constructor.

        Parameters
        ----------
        models : list of Class1ProcessingNeuralNetwork
            Neural networks in the ensemble.
        manifest_df : pandas.DataFrame
            Manifest dataframe. If not specified a new one will be created when
            needed.
        metadata_dataframes : dict of string -> pandas.DataFrame
            Arbitrary metadata associated with this predictor
        provenance_string : string, optional
            Optional info string to use in __str__.
        """
        self.models = models
        self._manifest_df = manifest_df
        self.metadata_dataframes = (
            dict(metadata_dataframes) if metadata_dataframes else {})
        self.provenance_string = provenance_string

    @property
    def sequence_lengths(self):
        """
        Supported maximum sequence lengths.

        Passing a peptide greater than the maximum supported length results
        in an error.

        Passing an N- or C-flank sequence greater than the maximum supported
        length results in some part of it being ignored.

        Returns
        -------
        dict of string -> int

        Keys are "peptide", "n_flank", "c_flank". Values give the maximum
        supported sequence length.
        """
        df = pandas.DataFrame([model.sequence_lengths for model in self.models])
        return {
            "peptide": df.peptide.min(),  # min: anything greater is error
            "n_flank": df.n_flank.max(),  # max: anything greater is ignored
            "c_flank": df.c_flank.max(),
        }

    def add_models(self, models):
        """
        Add models to the ensemble (in-place).

        Parameters
        ----------
        models : list of Class1ProcessingNeuralNetwork

        Returns
        -------
        list of string

        Names of the new models.
        """
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
            n_flanks=None,
            c_flanks=None,
            throw=True,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        """
        Predict antigen processing.

        Parameters
        ----------
        peptides : list of string
            Peptide sequences
        n_flanks : list of string
            Upstream sequence before each peptide
        c_flanks : list of string
            Downstream sequence after each peptide
        throw : boolean
            If True, a ValueError will be raised in the case of unsupported
            peptides. If False, a warning will be logged and the predictions
            for those peptides will be NaN.
        batch_size : int
            Prediction keras batch size.

        Returns
        -------
        numpy.array

        Processing scores. Range is 0-1, higher indicates more favorable
        processing.
        """
        return self.predict_to_dataframe(
            peptides=peptides,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            throw=throw,
            batch_size=batch_size).score.values

    def predict_to_dataframe(
            self,
            peptides,
            n_flanks=None,
            c_flanks=None,
            throw=True,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        """
        Predict antigen processing.

        See `predict` method for parameter descriptions.

        Returns
        -------
        pandas.DataFrame

        Processing predictions are in the "score" column. Also includes
        peptides and flanking sequences.
        """

        if n_flanks is None:
            n_flanks = [""] * len(peptides)
        if c_flanks is None:
            c_flanks = [""] * len(peptides)

        sequences = FlankingEncoding(
            peptides=peptides, n_flanks=n_flanks, c_flanks=c_flanks)
        return self.predict_to_dataframe_encoded(
            sequences=sequences, throw=throw, batch_size=batch_size)

    def predict_to_dataframe_encoded(
            self, sequences, throw=True, batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        """
        Predict antigen processing.

        See `predict` method for more information.

        Parameters
        ----------
        sequences : FlankingEncoding
        batch_size : int
        throw : boolean

        Returns
        -------
        pandas.DataFrame
        """

        score_array = []

        for (i, network) in enumerate(self.models):
            predictions = network.predict_encoded(
                sequences, throw=throw, batch_size=batch_size)
            score_array.append(predictions)

        score_array = numpy.array(score_array)

        result_df = pandas.DataFrame({
            "peptide": sequences.dataframe.peptide,
            "n_flank": sequences.dataframe.n_flank,
            "c_flank": sequences.dataframe.c_flank,
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
        the configurations of each Class1ProcessingNeuralNetwork, along with
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
        `Class1ProcessingPredictor` instance
        """
        if models_dir is None:
            models_dir = get_default_class1_processing_models_dir()

        manifest_path = join(models_dir, "manifest.csv")
        manifest_df = pandas.read_csv(manifest_path, nrows=max_models)

        models = []
        for (_, row) in manifest_df.iterrows():
            weights_filename = cls.weights_path(models_dir, row.model_name)
            config = json.loads(row.config_json)
            model = Class1ProcessingNeuralNetwork.from_config(
                config,
                weights=load_weights(abspath(weights_filename)))
            models.append(model)

        manifest_df["model"] = models

        logging.info("Loaded %d class1 processing models", len(models))

        provenance_string = None
        try:
            info_path = join(models_dir, "info.txt")
            info = pandas.read_csv(
                info_path, sep="\t", header=None, index_col=0).iloc[
                :, 0
            ].to_dict()
            provenance_string = "generated on %s" % info["trained on"]
        except OSError:
            pass

        result = cls(
            models=models,
            manifest_df=manifest_df,
            provenance_string=provenance_string)
        return result

    def __repr__(self):
        pieces = ["at 0x%0x" % id(self), "[mhcflurry %s]" % __version__]
        if self.provenance_string:
            pieces.append(self.provenance_string)
        return "<Class1ProcessingPredictor %s>" % " ".join(pieces)
