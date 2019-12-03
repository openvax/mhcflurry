from __future__ import print_function

from os.path import join, exists, abspath
from os import mkdir, environ
from socket import gethostname
from getpass import getuser

import time
import collections
import json
import hashlib
import logging
from six import string_types


import numpy
import pandas

import mhcnames

from .hyperparameters import HyperparameterDefaults
from .version import __version__
from .class1_neural_network import Class1NeuralNetwork, DEFAULT_PREDICT_BATCH_SIZE
from .encodable_sequences import EncodableSequences
from .regression_target import from_ic50, to_ic50
from .random_negative_peptides import RandomNegativePeptides
from .allele_encoding import MultipleAlleleEncoding, AlleleEncoding
from .auxiliary_input import AuxiliaryInputEncoder
from .batch_generator import MultiallelicMassSpecBatchGenerator
from .custom_loss import (
    MSEWithInequalities,
    MultiallelicMassSpecLoss,
    ZeroLoss)


class Class1PresentationPredictor(object):
    def __init__(
            self,
            class1_presentation_neural_networks,
            allele_to_sequence,
            manifest_df=None,
            metadata_dataframes=None):
        self.networks = class1_presentation_neural_networks
        self.allele_to_sequence = allele_to_sequence
        self._manifest_df = manifest_df
        self.metadata_dataframes = (
            dict(metadata_dataframes) if metadata_dataframes else {})

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
            for (i, model) in enumerate(self.networks):
                rows.append((
                    self.model_name(i),
                    json.dumps(model.get_config()),
                    model
                ))
            self._manifest_df = pandas.DataFrame(
                rows,
                columns=["model_name", "config_json", "model"])
        return self._manifest_df

    @property
    def max_alleles(self):
        max_alleles = self.networks[0].hyperparameters['max_alleles']
        assert all(
            n.hyperparameters['max_alleles'] == self.max_alleles
            for n in self.networks)
        return max_alleles

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
        return "LIGANDOME-CLASSI-%d-%s" % (
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

    def predict(self, peptides, alleles, batch_size=DEFAULT_PREDICT_BATCH_SIZE):
        return self.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            batch_size=batch_size).score.values

    def predict_to_dataframe(
            self,
            peptides,
            alleles,
            include_details=False,
            batch_size=DEFAULT_PREDICT_BATCH_SIZE):

        if isinstance(peptides, string_types):
            raise TypeError("peptides must be a list or array, not a string")
        if isinstance(alleles, string_types):
            raise TypeError(
                "alleles must be an iterable or MultipleAlleleEncoding")

        peptides = EncodableSequences.create(peptides)

        if not isinstance(alleles, MultipleAlleleEncoding):
            if len(alleles) > self.max_alleles:
                raise ValueError(
                    "When alleles is a list, it must have at most %d elements. "
                    "These alleles are taken to be a genotype for an "
                    "individual, and the strongest prediction across alleles "
                    "will be taken for each peptide. Note that this differs "
                    "from Class1AffinityPredictor.predict(), where alleles "
                    "is expected to be the same length as peptides."
                    % (
                        self.max_alleles))
            alleles = MultipleAlleleEncoding(
                experiment_names=numpy.tile("experiment", len(peptides)),
                experiment_to_allele_list={
                    "experiment": [
                        mhcnames.normalize_allele_name(a) for a in alleles
                    ],
                },
                allele_to_sequence=self.allele_to_sequence,
                max_alleles_per_experiment=self.max_alleles)

        score_array = []
        affinity_array = []

        for (i, network) in enumerate(self.networks):
            predictions = network.predict(
                peptides=peptides,
                allele_encoding=alleles,
                batch_size=batch_size)
            score_array.append(predictions.score)
            affinity_array.append(predictions.affinity)

        score_array = numpy.array(score_array)
        affinity_array = numpy.array(affinity_array)

        ensemble_scores = numpy.mean(score_array, axis=0)
        ensemble_affinity = numpy.mean(affinity_array, axis=0)
        top_allele_index = numpy.argmax(ensemble_scores, axis=-1)
        top_score = ensemble_scores[top_allele_index]
        top_affinity = ensemble_affinity[top_allele_index]

        result_df = pandas.DataFrame({"peptide": peptides.sequences})
        result_df["allele"] = alleles.alleles[top_allele_index]
        result_df["score"] = top_score
        result_df["affinity"] = to_ic50(top_affinity)

        if include_details:
            for i in range(self.max_alleles):
                result_df["allele%d" % (i + 1)] = alleles.allele[:, i]
                result_df["allele%d score" % (i + 1)] = ensemble_scores[:, i]
                result_df["allele%d score low" % (i + 1)] = numpy.percentile(
                    score_array[:, :, i], 5.0, axis=0)
                result_df["allele%d score high" % (i + 1)] = numpy.percentile(
                    score_array[:, :, i], 95.0, axis=0)
                result_df["allele%d affinity" % (i + 1)] = to_ic50(
                    ensemble_affinity[:, i])
                result_df["allele%d affinity low" % (i + 1)] = to_ic50(
                    numpy.percentile(affinity_array[:, :, i], 95.0, axis=0))
                result_df["allele%d affinity high" % (i + 1)] = to_ic50(
                    numpy.percentile(affinity_array[:, :, i], 5.0, axis=0))
        return result_df

    @staticmethod
    def save_weights(weights_list, filename):
        """
        Save the model weights to the given filename using numpy's ".npz"
        format.

        Parameters
        ----------
        weights_list : list of array

        filename : string
            Should end in ".npz".

        """
        numpy.savez(
            filename,
            **dict((("array_%d" % i), w) for (i, w) in enumerate(weights_list)))

    def check_consistency(self):
        """
        Verify that self.manifest_df is consistent with instance variables.

        Currently only checks for agreement on the total number of models.

        Throws AssertionError if inconsistent.
        """
        assert len(self.manifest_df) == len(self.networks), (
            "Manifest seems out of sync with models: %d vs %d entries: \n%s"% (
                len(self.manifest_df),
                len(self.networks),
                str(self.manifest_df)))

    def save(self, models_dir, model_names_to_write=None, write_metadata=True):
        """
        Serialize the predictor to a directory on disk. If the directory does
        not exist it will be created.

        The serialization format consists of a file called "manifest.csv" with
        the configurations of each Class1PresentationNeuralNetwork, along with
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
        # for example due to changes to the allele representation layer.
        # So we update the JSON configs here also.
        updated_network_config_jsons = []
        for (_, row) in sub_manifest_df.iterrows():
            updated_network_config_jsons.append(
                json.dumps(row.model.get_config()))
            weights_path = self.weights_path(models_dir, row.model_name)
            self.save_weights(
                row.model.get_weights(), weights_path)
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

        # Save allele sequences
        if self.allele_to_sequence is not None:
            allele_to_sequence_df = pandas.DataFrame(
                list(self.allele_to_sequence.items()),
                columns=['allele', 'sequence']
            )
            allele_to_sequence_df.to_csv(
                join(models_dir, "allele_sequences.csv"), index=False)
            logging.info("Wrote: %s", join(models_dir, "allele_sequences.csv"))

    @staticmethod
    def load(models_dir=None, max_models=None):
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
        `Class1AffinityPredictor` instance
        """
        if models_dir is None:
            models_dir = get_default_class1_models_dir()

        manifest_path = join(models_dir, "manifest.csv")
        manifest_df = pandas.read_csv(manifest_path, nrows=max_models)

        allele_to_allele_specific_models = collections.defaultdict(list)
        class1_pan_allele_models = []
        all_models = []
        for (_, row) in manifest_df.iterrows():
            weights_filename = Class1AffinityPredictor.weights_path(
                models_dir, row.model_name)
            config = json.loads(row.config_json)

            # We will lazy-load weights when the network is used.
            model = Class1NeuralNetwork.from_config(
                config,
                weights_loader=partial(
                    Class1AffinityPredictor.load_weights,
                    abspath(weights_filename)))
            if row.allele == "pan-class1":
                class1_pan_allele_models.append(model)
            else:
                allele_to_allele_specific_models[row.allele].append(model)
            all_models.append(model)

        manifest_df["model"] = all_models

        # Load allele sequences
        allele_to_sequence = None
        if exists(join(models_dir, "allele_sequences.csv")):
            allele_to_sequence = pandas.read_csv(
                join(models_dir, "allele_sequences.csv"),
                index_col=0).iloc[:, 0].to_dict()

        allele_to_percent_rank_transform = {}
        percent_ranks_path = join(models_dir, "percent_ranks.csv")
        if exists(percent_ranks_path):
            percent_ranks_df = pandas.read_csv(percent_ranks_path, index_col=0)
            for allele in percent_ranks_df.columns:
                allele_to_percent_rank_transform[allele] = (
                    PercentRankTransform.from_series(percent_ranks_df[allele]))

        logging.info(
            "Loaded %d class1 pan allele predictors, %d allele sequences, "
            "%d percent rank distributions, and %d allele specific models: %s",
            len(class1_pan_allele_models),
            len(allele_to_sequence) if allele_to_sequence else 0,
            len(allele_to_percent_rank_transform),
            sum(len(v) for v in allele_to_allele_specific_models.values()),
            ", ".join(
                "%s (%d)" % (allele, len(v))
                for (allele, v)
                in sorted(allele_to_allele_specific_models.items())))

        result = Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_sequence=allele_to_sequence,
            manifest_df=manifest_df,
            allele_to_percent_rank_transform=allele_to_percent_rank_transform,
        )
        if optimization_level >= 1:
            optimized = result.optimize()
            logging.info(
                "Model optimization %s",
                "succeeded" if optimized else "not supported for these models")
        return result



    # TODO: implement saving and loading