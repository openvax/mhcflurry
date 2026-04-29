import collections
import hashlib
import json
import logging
import time
import warnings
from os.path import join, exists, abspath
from os import mkdir, environ
from socket import gethostname
from getpass import getuser
from functools import partial
import numpy
import pandas


from .class1_neural_network import Class1NeuralNetwork
from .common import (
    random_peptides,
    positional_frequency_matrix,
    normalize_allele_name
)
from .downloads import get_default_class1_models_dir
from .encodable_sequences import EncodableSequences
from .percent_rank_transform import PercentRankTransform
from .regression_target import to_ic50
from .version import __version__
from .ensemble_centrality import CENTRALITY_MEASURES
from .allele_encoding import AlleleEncoding
from .common import save_weights, load_weights


# Default function for combining predictions across models in an ensemble.
# See ensemble_centrality.py for other options.
DEFAULT_CENTRALITY_MEASURE = "mean"

# Any value > 0 will result in attempting to optimize models after loading.
OPTIMIZATION_LEVEL = int(environ.get("MHCFLURRY_OPTIMIZATION_LEVEL", 1))


class Class1AffinityPredictor(object):
    """
    High-level interface for peptide/MHC I binding affinity prediction.

    This class manages low-level `Class1NeuralNetwork` instances, each of which
    wraps a single PyTorch network. The purpose of `Class1AffinityPredictor` is to
    implement ensembles, handling of multiple alleles, and predictor loading and
    saving. It also provides a place to keep track of metadata like prediction
    histograms for percentile rank calibration.
    """
    def __init__(
            self,
            allele_to_allele_specific_models=None,
            class1_pan_allele_models=None,
            allele_to_sequence=None,
            manifest_df=None,
            allele_to_percent_rank_transform=None,
            metadata_dataframes=None,
            provenance_string=None,
            optimization_info=None):
        """
        Parameters
        ----------
        allele_to_allele_specific_models : dict of string -> list of `Class1NeuralNetwork`
            Ensemble of single-allele models to use for each allele.

        class1_pan_allele_models : list of `Class1NeuralNetwork`
            Ensemble of pan-allele models.

        allele_to_sequence : dict of string -> string
            MHC allele name to fixed-length amino acid sequence (sometimes
            referred to as the pseudosequence). Required only if
            class1_pan_allele_models is specified.

        manifest_df : `pandas.DataFrame`, optional
            Must have columns: model_name, allele, config_json, model.
            Only required if you want to update an existing serialization of a
            Class1AffinityPredictor. Otherwise this dataframe will be generated
            automatically based on the supplied models.

        allele_to_percent_rank_transform : dict of string -> `PercentRankTransform`, optional
            `PercentRankTransform` instances to use for each allele

        metadata_dataframes : dict of string -> pandas.DataFrame, optional
            Optional additional dataframes to write to the models dir when
            save() is called. Useful for tracking provenance.

        provenance_string : string, optional
            Optional info string to use in __str__.

        optimization_info : dict, optional
            Dict describing any optimizations already performed on the model.
            The only currently supported optimization is to merge ensembles
            together into one PyTorch model.
        """

        if allele_to_allele_specific_models is None:
            allele_to_allele_specific_models = {}
        if class1_pan_allele_models is None:
            class1_pan_allele_models = []

        self.allele_to_sequence = (
            dict(allele_to_sequence)
            if allele_to_sequence is not None else None)  # make a copy

        self._master_allele_encoding = None
        if class1_pan_allele_models:
            assert self.allele_to_sequence

        self.allele_to_allele_specific_models = allele_to_allele_specific_models
        self.class1_pan_allele_models = class1_pan_allele_models
        self._manifest_df = manifest_df

        if not allele_to_percent_rank_transform:
            allele_to_percent_rank_transform = {}
        self.allele_to_percent_rank_transform = allele_to_percent_rank_transform
        self.metadata_dataframes = (
            dict(metadata_dataframes) if metadata_dataframes else {})
        self._cache = {}
        self.optimization_info = optimization_info if optimization_info else {}

        assert isinstance(self.allele_to_allele_specific_models, dict)
        assert isinstance(self.class1_pan_allele_models, list)

        self.provenance_string = provenance_string
        self.allele_to_canonical = {}  # populated by load()

    @property
    def manifest_df(self):
        """
        A pandas.DataFrame describing the models included in this predictor.

        Based on:
        - self.class1_pan_allele_models
        - self.allele_to_allele_specific_models

        Returns
        -------
        pandas.DataFrame
        """
        if self._manifest_df is None:
            rows = []
            for (i, model) in enumerate(self.class1_pan_allele_models):
                rows.append((
                    self.model_name("pan-class1", i),
                    "pan-class1",
                    json.dumps(model.get_config()),
                    model
                ))
            for (allele, models) in self.allele_to_allele_specific_models.items():
                for (i, model) in enumerate(models):
                    rows.append((
                        self.model_name(allele, i),
                        allele,
                        json.dumps(model.get_config()),
                        model
                    ))
            self._manifest_df = pandas.DataFrame(
                rows,
                columns=["model_name", "allele", "config_json", "model"])
        return self._manifest_df

    def clear_cache(self):
        """
        Clear values cached based on the neural networks in this predictor.

        Users should call this after mutating any of the following:
            - self.class1_pan_allele_models
            - self.allele_to_allele_specific_models
            - self.allele_to_sequence

        Methods that mutate these instance variables will call this method on
        their own if needed.
        """
        self._cache.clear()
        self.provenance_string = None

    @property
    def neural_networks(self):
        """
        List of the neural networks in the ensemble.

        Returns
        -------
        list of `Class1NeuralNetwork`
        """
        result = []
        for models in self.allele_to_allele_specific_models.values():
            result.extend(models)
        result.extend(self.class1_pan_allele_models)
        return result

    @classmethod
    def merge(cls, predictors):
        """
        Merge the ensembles of two or more `Class1AffinityPredictor` instances.

        Note: the resulting merged predictor will NOT have calibrated percentile
        ranks. Call `calibrate_percentile_ranks` on it if these are needed.

        Parameters
        ----------
        predictors : sequence of `Class1AffinityPredictor`

        Returns
        -------
        `Class1AffinityPredictor` instance

        """
        assert len(predictors) > 0
        if len(predictors) == 1:
            return predictors[0]

        allele_to_allele_specific_models = collections.defaultdict(list)
        class1_pan_allele_models = []
        allele_to_sequence = predictors[0].allele_to_sequence

        for predictor in predictors:
            for (allele, networks) in (
                    predictor.allele_to_allele_specific_models.items()):
                allele_to_allele_specific_models[allele].extend(networks)
            class1_pan_allele_models.extend(
                predictor.class1_pan_allele_models)

        return Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_sequence=allele_to_sequence
        )

    def merge_in_place(self, others):
        """
        Add the models present in other predictors into the current predictor.

        Parameters
        ----------
        others : list of Class1AffinityPredictor
            Other predictors to merge into the current predictor.

        Returns
        -------
        list of string : names of newly added models
        """
        new_model_names = []
        original_manifest = self.manifest_df
        new_manifest_rows = []
        for predictor in others:
            for model in predictor.class1_pan_allele_models:
                model_name = self.model_name(
                    "pan-class1",
                    len(self.class1_pan_allele_models))
                row = pandas.Series(collections.OrderedDict([
                    ("model_name", model_name),
                    ("allele", "pan-class1"),
                    ("config_json", json.dumps(model.get_config())),
                    ("model", model),
                ])).to_frame().T
                new_manifest_rows.append(row)
                self.class1_pan_allele_models.append(model)
                new_model_names.append(model_name)

            for allele in predictor.allele_to_allele_specific_models:
                if allele not in self.allele_to_allele_specific_models:
                    self.allele_to_allele_specific_models[allele] = []
                current_models = self.allele_to_allele_specific_models[allele]
                for model in predictor.allele_to_allele_specific_models[allele]:
                    model_name = self.model_name(allele, len(current_models))
                    row = pandas.Series(collections.OrderedDict([
                        ("model_name", model_name),
                        ("allele", allele),
                        ("config_json", json.dumps(model.get_config())),
                        ("model", model),
                    ])).to_frame().T
                    new_manifest_rows.append(row)
                    current_models.append(model)
                    new_model_names.append(model_name)

        self._manifest_df = pandas.concat(
            [original_manifest] + new_manifest_rows,
            ignore_index=True)

        self.clear_cache()
        self.check_consistency()
        return new_model_names

    def canonicalize_allele_name(self, raw_name):
        """
        Normalize an allele name and map it to the canonical pseudosequence
        key if possible.

        Tries without IMGT aliases first so that alleles like HLA-C*01:01
        (which aliases map to C*01:02) resolve to their own pseudosequence
        when one exists.

        Parameters
        ----------
        raw_name : str

        Returns
        -------
        str
        """
        # Try without aliases first — this matches pseudosequence keys
        # directly and avoids mhcgnomes alias remapping or Q/N annotations.
        if self.allele_to_sequence:
            no_alias = normalize_allele_name(
                raw_name, raise_on_error=False, use_allele_aliases=False)
            if no_alias is not None and no_alias in self.allele_to_sequence:
                return no_alias
        # Fall back to aliases and map through canonical lookup.
        normalized = normalize_allele_name(raw_name)
        return self.allele_to_canonical.get(normalized, normalized)

    @property
    def supported_alleles(self):
        """
        Alleles for which predictions can be made.

        Returns
        -------
        list of string
        """
        if 'supported_alleles' not in self._cache:
            result = set(self.allele_to_allele_specific_models)
            if self.allele_to_sequence:
                result = result.union(self.allele_to_sequence)
            self._cache["supported_alleles"] = sorted(result)
        return self._cache["supported_alleles"]

    @property
    def supported_peptide_lengths(self):
        """
        (minimum, maximum) lengths of peptides supported by *all models*,
        inclusive.

        Returns
        -------
        (int, int) tuple

        """
        if 'supported_peptide_lengths' not in self._cache:
            length_ranges = set(
                network.supported_peptide_lengths
                for network in self.neural_networks)
            result = (
                max(lower for (lower, upper) in length_ranges),
                min(upper for (lower, upper) in length_ranges))
            self._cache["supported_peptide_lengths"] = result
        return self._cache["supported_peptide_lengths"]

    def check_consistency(self):
        """
        Verify that self.manifest_df is consistent with:
        - self.class1_pan_allele_models
        - self.allele_to_allele_specific_models

        Currently only checks for agreement on the total number of models.

        Throws AssertionError if inconsistent.
        """
        num_models = len(self.class1_pan_allele_models) + sum(
            len(v) for v in self.allele_to_allele_specific_models.values())
        assert len(self.manifest_df) == num_models, (
            "Manifest seems out of sync with models: %d vs %d entries: "
            "\n%s\npan-allele: %s\nallele-specific: %s"% (
                len(self.manifest_df),
                num_models,
                str(self.manifest_df),
                str(self.class1_pan_allele_models),
                str(self.allele_to_allele_specific_models)))

    def save(self, models_dir, model_names_to_write=None, write_metadata=True):
        """
        Serialize the predictor to a directory on disk. If the directory does
        not exist it will be created.

        The serialization format consists of a file called "manifest.csv" with
        the configurations of each Class1NeuralNetwork, along with per-network
        files giving the model weights. If there are pan-allele predictors in
        the ensemble, the allele sequences are also stored in the
        directory. There is also a small file "index.txt" with basic metadata:
        when the models were trained, by whom, on what host.

        Parameters
        ----------
        models_dir : string
            Path to directory. It will be created if it doesn't exist.

        model_names_to_write : list of string, optional
            Only write the weights for the specified models. Useful for
            incremental updates during training.

        write_metadata : boolean, optional
            Whether to write optional metadata
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

        # Save allele sequences
        if self.allele_to_sequence is not None:
            allele_to_sequence_df = pandas.DataFrame(
                list(self.allele_to_sequence.items()),
                columns=['allele', 'sequence']
            )
            allele_to_sequence_df.to_csv(
                join(models_dir, "allele_sequences.csv"), index=False)
            logging.info("Wrote: %s", join(models_dir, "allele_sequences.csv"))

        if self.allele_to_percent_rank_transform:
            percent_ranks_df = None
            for (allele, transform) in self.allele_to_percent_rank_transform.items():
                series = transform.to_series()
                if percent_ranks_df is None:
                    percent_ranks_df = {}
                    percent_ranks_df_index = series.index
                numpy.testing.assert_array_almost_equal(
                    series.index.values,
                    percent_ranks_df_index.values)
                percent_ranks_df[allele] = series.values
            percent_ranks_df = pandas.DataFrame(
                percent_ranks_df,
                index=percent_ranks_df_index)
            percent_ranks_path = join(models_dir, "percent_ranks.csv")
            percent_ranks_df.to_csv(
                percent_ranks_path,
                index=True,
                index_label="bin")
            logging.info("Wrote: %s", percent_ranks_path)

        if self.optimization_info:
            # If the model being saved was optimized, we need to save that
            # information since it can affect how predictions are performed
            # (e.g. stitched-together ensembles output concatenated results,
            # which then need to be averaged outside the model).
            optimization_info_path = join(models_dir, "optimization_info.json")
            with open(optimization_info_path, "w") as fd:
                json.dump(self.optimization_info, fd, indent=4)

    @staticmethod
    def load(models_dir=None, max_models=None, optimization_level=None):
        """
        Deserialize a predictor from a directory on disk.

        Parameters
        ----------
        models_dir : string
            Path to directory. If unspecified the default downloaded models are
            used.

        max_models : int, optional
            Maximum number of `Class1NeuralNetwork` instances to load

        optimization_level : int
            If >0, model optimization will be attempted. Defaults to value of
            environment variable MHCFLURRY_OPTIMIZATION_LEVEL.

        Returns
        -------
        `Class1AffinityPredictor` instance
        """
        if models_dir is None:
            try:
                models_dir = get_default_class1_models_dir()
            except RuntimeError as e:
                # Fall back to the affinity predictor included in presentation
                # predictor if possible.
                from mhcflurry.class1_presentation_predictor import (
                    Class1PresentationPredictor)
                try:
                    presentation_predictor = Class1PresentationPredictor.load()
                    return presentation_predictor.affinity_predictor
                except RuntimeError:
                    raise e

        if optimization_level is None:
            optimization_level = OPTIMIZATION_LEVEL

        manifest_path = join(models_dir, "manifest.csv")
        manifest_df = pandas.read_csv(manifest_path, nrows=max_models)

        # ----- Load pseudosequences first so we can canonicalize -----
        allele_to_sequence = None
        allele_to_canonical = {}
        if exists(join(models_dir, "allele_sequences.csv")):
            allele_to_sequence = pandas.read_csv(
                join(models_dir, "allele_sequences.csv"),
                index_col=0).iloc[:, 0].to_dict()

            # Re-normalize allele names. We first try without IMGT allele
            # aliases to preserve current nomenclature. If the parse fails
            # or the pseudosequence contains unknown (X) positions, we
            # retry with aliases — retired allele names like B*44:01 (an
            # IMGT error reassigned to B*44:02 in 1994) often have
            # incomplete pseudosequences, and the alias target may have a
            # complete one. If mhcgnomes can't parse either way, keep the
            # raw name so the pseudosequence remains available.
            renormalized = {}
            skipped_non_class1 = []
            for (name, value) in allele_to_sequence.items():
                normalized = normalize_allele_name(
                    name, raise_on_error=False, use_allele_aliases=False)
                if normalized is None or "X" in value:
                    alias_normalized = normalize_allele_name(
                        name, raise_on_error=False, use_allele_aliases=True)
                    if alias_normalized is not None:
                        normalized = alias_normalized
                if normalized is None:
                    # Detect class II, TAP, and pseudogene entries —
                    # these don't belong in a class I predictor and
                    # always have incomplete pseudosequences.
                    gene = name.split("*")[0].split("-")[-1] if "-" in name else ""
                    if ("X" in value and
                            any(tag in gene
                                for tag in ("DAA", "DAB", "TAP", "PS"))):
                        skipped_non_class1.append(name)
                        continue
                    normalized = name
                if normalized in renormalized and name != normalized:
                    existing = renormalized[normalized]
                    if value.count("X") < existing.count("X"):
                        renormalized[normalized] = value
                    continue
                renormalized[normalized] = value
            allele_to_sequence = renormalized
            if skipped_non_class1:
                logging.info(
                    "Skipped %d non-class-I entries from pseudosequence "
                    "file (class II / TAP / pseudogene with incomplete "
                    "pseudosequences): %s",
                    len(skipped_non_class1),
                    ", ".join(sorted(skipped_non_class1)[:10])
                    + (" ..." if len(skipped_non_class1) > 10 else ""))

            # Map mhcgnomes-aliased forms back to pseudosequence keys.
            # e.g. Mamu-A1*007:01 -> Mamu-A*07:01
            for canonical_name in allele_to_sequence:
                aliased = normalize_allele_name(
                    canonical_name, raise_on_error=False,
                    use_allele_aliases=True)
                if (aliased is not None and aliased != canonical_name
                        and aliased not in allele_to_sequence):
                    allele_to_canonical[aliased] = canonical_name

        def to_canonical(raw_name):
            """Normalize a raw allele name to its canonical pseudosequence key."""
            n = normalize_allele_name(raw_name, raise_on_error=False) or raw_name
            return allele_to_canonical.get(n, n)

        # ----- Load manifest -----
        allele_to_allele_specific_models = collections.defaultdict(list)
        class1_pan_allele_models = []
        all_models = []
        for (_, row) in manifest_df.iterrows():
            weights_filename = Class1AffinityPredictor.weights_path(
                models_dir, row.model_name)
            config = json.loads(row.config_json)

            model = Class1NeuralNetwork.from_config(
                config,
                weights_loader=partial(load_weights, abspath(weights_filename)))
            if row.allele == "pan-class1":
                class1_pan_allele_models.append(model)
            else:
                allele_to_allele_specific_models[
                    to_canonical(row.allele)].append(model)
            all_models.append(model)

        manifest_df["model"] = all_models

        # ----- Load percent ranks -----
        allele_to_percent_rank_transform = {}
        percent_ranks_path = join(models_dir, "percent_ranks.csv")
        if exists(percent_ranks_path):
            percent_ranks_df = pandas.read_csv(percent_ranks_path, index_col=0)
            for allele in percent_ranks_df.columns:
                canonical = to_canonical(allele)
                if (canonical in allele_to_percent_rank_transform and
                        allele != canonical):
                    continue
                allele_to_percent_rank_transform[canonical] = (
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

        optimization_info = None
        try:
            optimization_info_path = join(models_dir, "optimization_info.json")
            with open(optimization_info_path) as fd:
                optimization_info = json.load(fd)
        except OSError:
            pass

        result = Class1AffinityPredictor(
            allele_to_allele_specific_models=allele_to_allele_specific_models,
            class1_pan_allele_models=class1_pan_allele_models,
            allele_to_sequence=allele_to_sequence,
            manifest_df=manifest_df,
            allele_to_percent_rank_transform=allele_to_percent_rank_transform,
            provenance_string=provenance_string,
            optimization_info=optimization_info,
        )
        if allele_to_sequence is not None:
            result.allele_to_canonical = allele_to_canonical
        if optimization_level >= 1:
            optimized = result.optimize()
            logging.info(
                "Model optimization %s",
                "succeeded" if optimized else "not supported for these models")
        return result

    def __repr__(self):
        pieces = ["at 0x%0x" % id(self), "[mhcflurry %s]" % __version__]

        pan_models = len(self.class1_pan_allele_models)
        total_models = len(self.neural_networks)
        if total_models == 0:
            pieces.append("[empty]")
        elif pan_models == total_models:
            pieces.append("[pan]")
        elif pan_models == 0:
            pieces.append("[allele-specific]")
        else:
            pieces.append("[pan+allele-specific]")

        if self.provenance_string:
            pieces.append(self.provenance_string)

        return "<Class1AffinityPredictor %s>" % " ".join(pieces)

    def optimize(self, warn=True):
        """
        EXPERIMENTAL: Optimize the predictor for faster predictions.

        Currently the only optimization implemented is to merge multiple pan-
        allele predictors at the PyTorch level.

        The optimization is performed in-place, mutating the instance.

        Returns
        ----------
        bool
            Whether optimization was performed

        """
        num_class1_pan_allele_models = len(self.class1_pan_allele_models)
        if num_class1_pan_allele_models > 1:
            provenance_string = self.provenance_string
            try:
                self.class1_pan_allele_models = [
                    Class1NeuralNetwork.merge(
                        self.class1_pan_allele_models,
                        merge_method="concatenate")
                ]
            except NotImplementedError as e:
                if warn:
                    logging.warning("Optimization failed: %s", str(e))
                return False
            self._manifest_df = None
            self.clear_cache()
            self.optimization_info["pan_models_merged"] = True
            self.optimization_info["num_pan_models_merged"] = (
                num_class1_pan_allele_models)
            self.provenance_string = provenance_string
        else:
            return False
        return True

    @staticmethod
    def model_name(allele, num):
        """
        Generate a model name

        Parameters
        ----------
        allele : string
        num : int

        Returns
        -------
        string

        """
        random_string = hashlib.sha1(
            str(time.time()).encode()).hexdigest()[:16]
        return "%s-%d-%s" % (
            allele.upper().replace("*", "_").replace(":", "_"),
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

    @property
    def master_allele_encoding(self):
        """
        An AlleleEncoding containing the universe of alleles specified by
        self.allele_to_sequence.

        Returns
        -------
        AlleleEncoding
        """
        if (self._master_allele_encoding is None or
                self._master_allele_encoding.allele_to_sequence !=
                self.allele_to_sequence):
            self._master_allele_encoding = AlleleEncoding(
                allele_to_sequence=self.allele_to_sequence)
        return self._master_allele_encoding

    def fit_allele_specific_predictors(
            self,
            n_models,
            architecture_hyperparameters_list,
            allele,
            peptides,
            affinities,
            inequalities=None,
            train_rounds=None,
            models_dir_for_save=None,
            verbose=0,
            progress_preamble="",
            progress_print_interval=5.0):
        """
        Fit one or more allele specific predictors for a single allele using one
        or more neural network architectures.

        The new predictors are saved in the Class1AffinityPredictor instance
        and will be used on subsequent calls to `predict`.

        Parameters
        ----------
        n_models : int
            Number of neural networks to fit

        architecture_hyperparameters_list : list of dict
            List of hyperparameter sets.

        allele : string

        peptides : `EncodableSequences` or list of string

        affinities : list of float
            nM affinities

        inequalities : list of string, each element one of ">", "<", or "="
            See `Class1NeuralNetwork.fit` for details.

        train_rounds : sequence of int
            Each training point i will be used on training rounds r for which
            train_rounds[i] > r, r >= 0.

        models_dir_for_save : string, optional
            If specified, the Class1AffinityPredictor is (incrementally) written
            to the given models dir after each neural network is fit.

        verbose : int
            Verbosity level for training output

        progress_preamble : string
            Optional string of information to include in each progress update

        progress_print_interval : float
            How often (in seconds) to print progress. Set to None to disable.

        Returns
        -------
        list of `Class1NeuralNetwork`
        """

        allele = normalize_allele_name(allele)
        if allele not in self.allele_to_allele_specific_models:
            self.allele_to_allele_specific_models[allele] = []

        encodable_peptides = EncodableSequences.create(peptides)
        peptides_affinities_inequalities_per_round = [
            (encodable_peptides, affinities, inequalities)
        ]

        if train_rounds is not None:
            for round in sorted(set(train_rounds)):
                round_mask = train_rounds > round
                if round_mask.any():
                    sub_encodable_peptides = EncodableSequences.create(
                        encodable_peptides.sequences[round_mask])
                    peptides_affinities_inequalities_per_round.append((
                        sub_encodable_peptides,
                        affinities[round_mask],
                        None if inequalities is None else inequalities[round_mask]))
        n_rounds = len(peptides_affinities_inequalities_per_round)

        n_architectures = len(architecture_hyperparameters_list)

        # Adjust progress info to indicate number of models and
        # architectures.
        pieces = []
        if n_models > 1:
            pieces.append("Model {model_num:2d} / {n_models:2d}")
        if n_architectures > 1:
            pieces.append(
                "Architecture {architecture_num:2d} / {n_architectures:2d}")
        if len(peptides_affinities_inequalities_per_round) > 1:
            pieces.append("Round {round:2d} / {n_rounds:2d}")
        pieces.append("{n_peptides:4d} peptides")
        progress_preamble_template = "[ %s ] {user_progress_preamble}" % (
            ", ".join(pieces))

        models = []
        for model_num in range(n_models):
            for (architecture_num, architecture_hyperparameters) in enumerate(
                    architecture_hyperparameters_list):
                model = Class1NeuralNetwork(**architecture_hyperparameters)
                for round_num in range(n_rounds):
                    (round_peptides, round_affinities, round_inequalities) = (
                        peptides_affinities_inequalities_per_round[round_num]
                    )
                    model.fit(
                        round_peptides,
                        round_affinities,
                        inequalities=round_inequalities,
                        verbose=verbose,
                        progress_preamble=progress_preamble_template.format(
                            n_peptides=len(round_peptides),
                            round=round_num,
                            n_rounds=n_rounds,
                            user_progress_preamble=progress_preamble,
                            model_num=model_num + 1,
                            n_models=n_models,
                            architecture_num=architecture_num + 1,
                            n_architectures=n_architectures),
                        progress_print_interval=progress_print_interval)

                model_name = self.model_name(allele, model_num)
                row = pandas.Series(collections.OrderedDict([
                    ("model_name", model_name),
                    ("allele", allele),
                    ("config_json", json.dumps(model.get_config())),
                    ("model", model),
                ])).to_frame().T
                self._manifest_df = pandas.concat(
                    [self.manifest_df, row], ignore_index=True)
                self.allele_to_allele_specific_models[allele].append(model)
                if models_dir_for_save:
                    self.save(
                        models_dir_for_save, model_names_to_write=[model_name])
                models.append(model)

        self.clear_cache()
        return models

    def fit_class1_pan_allele_models(
            self,
            n_models,
            architecture_hyperparameters,
            alleles,
            peptides,
            affinities,
            inequalities,
            models_dir_for_save=None,
            verbose=1,
            progress_preamble="",
            progress_print_interval=5.0):
        """
        Fit one or more pan-allele predictors using a single neural network
        architecture.

        The new predictors are saved in the Class1AffinityPredictor instance
        and will be used on subsequent calls to `predict`.

        Parameters
        ----------
        n_models : int
            Number of neural networks to fit

        architecture_hyperparameters : dict

        alleles : list of string
            Allele names (not sequences) corresponding to each peptide

        peptides : `EncodableSequences` or list of string

        affinities : list of float
            nM affinities

        inequalities : list of string, each element one of ">", "<", or "="
            See Class1NeuralNetwork.fit for details.

        models_dir_for_save : string, optional
            If specified, the Class1AffinityPredictor is (incrementally) written
            to the given models dir after each neural network is fit.

        verbose : int
            Verbosity level for training output

        progress_preamble : string
            Optional string of information to include in each progress update

        progress_print_interval : float
            How often (in seconds) to print progress. Set to None to disable.

        Returns
        -------
        list of `Class1NeuralNetwork`
        """

        alleles = pandas.Series(alleles).map(normalize_allele_name)
        allele_encoding = AlleleEncoding(
            alleles,
            borrow_from=self.master_allele_encoding)

        encodable_peptides = EncodableSequences.create(peptides)
        models = []
        for i in range(n_models):
            logging.info("Training model %d / %d", i + 1, n_models)
            model = Class1NeuralNetwork(**architecture_hyperparameters)
            model.fit(
                encodable_peptides,
                affinities,
                inequalities=inequalities,
                allele_encoding=allele_encoding,
                verbose=verbose,
                progress_preamble=progress_preamble,
                progress_print_interval=progress_print_interval)

            model_name = self.model_name("pan-class1", i)
            row = pandas.Series(collections.OrderedDict([
                ("model_name", model_name),
                ("allele", "pan-class1"),
                ("config_json", json.dumps(model.get_config())),
                ("model", model),
            ])).to_frame().T
            self._manifest_df = pandas.concat(
                [self.manifest_df, row], ignore_index=True)
            self.class1_pan_allele_models.append(model)
            if models_dir_for_save:
                self.save(
                    models_dir_for_save, model_names_to_write=[model_name])
            models.append(model)

        self.clear_cache()
        return models

    def add_pan_allele_model(self, model, models_dir_for_save=None):
        """
        Add a pan-allele model to the ensemble and optionally do an incremental
        save.

        Parameters
        ----------
        model : Class1NeuralNetwork
        models_dir_for_save : string
            Directory to save resulting ensemble to
        """
        model_name = self.model_name("pan-class1", 1)
        row = pandas.Series(collections.OrderedDict([
            ("model_name", model_name),
            ("allele", "pan-class1"),
            ("config_json", json.dumps(model.get_config())),
            ("model", model),
        ])).to_frame().T
        self._manifest_df = pandas.concat(
            [self.manifest_df, row], ignore_index=True)
        self.class1_pan_allele_models.append(model)
        self.clear_cache()
        self.check_consistency()
        if models_dir_for_save:
            self.save(
                models_dir_for_save, model_names_to_write=[model_name])

    def percentile_ranks(self, affinities, allele=None, alleles=None, throw=True):
        """
        Return percentile ranks for the given ic50 affinities and alleles.

        The 'allele' and 'alleles' argument are as in the `predict` method.
        Specify one of these.

        Parameters
        ----------
        affinities : sequence of float
            nM affinities
        allele : string
        alleles : sequence of string
        throw : boolean
            If True, a ValueError will be raised in the case of unsupported
            alleles. If False, a warning will be logged and NaN will be returned
            for those percentile ranks.

        Returns
        -------
        numpy.array of float
        """
        if allele is not None:
            normalized_allele = self.canonicalize_allele_name(allele)
            try:
                transform = self.allele_to_percent_rank_transform[normalized_allele]
                return transform.transform(affinities)
            except KeyError:
                if self.allele_to_sequence:
                    # See if we have information for an equivalent allele
                    sequence = self.allele_to_sequence[normalized_allele]
                    other_alleles = [
                        other_allele for (other_allele, other_sequence)
                        in self.allele_to_sequence.items()
                        if other_sequence == sequence
                    ]
                    for other_allele in other_alleles:
                        if other_allele in self.allele_to_percent_rank_transform:
                            transform = self.allele_to_percent_rank_transform[
                                other_allele]
                            return transform.transform(affinities)

                msg = "Allele %s has no percentile rank information" % (
                    allele + (
                        "" if allele == normalized_allele
                        else " (normalized to %s)" % normalized_allele))
                if throw:
                    raise ValueError(msg)
                warnings.warn(msg)
                return numpy.ones(len(affinities)) * numpy.nan  # Return NaNs

        if alleles is None:
            raise ValueError("Specify allele or alleles")

        df = pandas.DataFrame({"affinity": affinities})
        df["allele"] = alleles
        df["result"] = numpy.nan
        for (allele, sub_df) in df.groupby("allele"):
            df.loc[sub_df.index, "result"] = self.percentile_ranks(
                sub_df.affinity, allele=allele, throw=throw)
        return df.result.values

    def _prepare_peptides_with_optional_cache(
            self, peptides, *, encoding_cache_dir=None):
        """Return an EncodableSequences for the predict path.

        If ``encoding_cache_dir`` is None: identical to
        ``EncodableSequences.create(peptides)`` — current behavior.

        If set, resolves the peptide_encoding params from the first
        neural network's hyperparameters, looks up / builds the cache
        entry for this peptide list, and returns an EncodableSequences
        with the per-instance ``encoding_cache`` prepopulated.
        Downstream calls to ``peptides_to_network_input`` in each network
        hit the prepopulated cache and skip the BLOSUM62 pass.

        When the ensemble's networks have heterogeneous
        ``peptide_encoding`` configs, only the networks matching the
        cache params benefit. Other networks fall through to direct
        encoding — their peptides_to_network_input call misses the
        cache and runs the encoder normally. No correctness impact.
        """
        if encoding_cache_dir is None:
            return EncodableSequences.create(peptides)

        # Need at least one network to read the encoding config from.
        if not self.neural_networks:
            return EncodableSequences.create(peptides)

        # Import locally so module-level import doesn't pull cache deps
        # into mhcflurry startup for callers who never use the cache.
        from .encoding_cache import (
            EncodingCache,
            EncodingParams,
            make_prepopulated_encodable_sequences,
        )

        cfg = self.neural_networks[0].hyperparameters.get(
            "peptide_encoding", {}
        )
        try:
            params = EncodingParams(**cfg)
        except TypeError:
            # Unknown kwargs (e.g. a new field we don't support) — fall
            # back to direct encoding rather than crash the predict call.
            logging.warning(
                "encoding_cache_dir was set but peptide_encoding %r "
                "contains kwargs not accepted by EncodingParams. Falling "
                "back to uncached encoding.", cfg
            )
            return EncodableSequences.create(peptides)

        # Normalize peptide list; EncodingCache hashes list order +
        # content, so consistent list type matters.
        if hasattr(peptides, "sequences"):
            peptide_list = list(peptides.sequences)
        else:
            peptide_list = list(peptides)

        if not peptide_list:
            return EncodableSequences.create(peptides)

        cache = EncodingCache(cache_dir=encoding_cache_dir, params=params)
        encoded, _peptide_to_idx = cache.get_or_build(peptide_list)
        return make_prepopulated_encodable_sequences(
            peptide_list, encoded, params
        )

    def predict(
            self,
            peptides,
            alleles=None,
            allele=None,
            throw=True,
            centrality_measure=DEFAULT_CENTRALITY_MEASURE,
            model_kwargs={},
            encoding_cache_dir=None):
        """
        Predict nM binding affinities.

        If multiple predictors are available for an allele, the predictions are
        the geometric means of the individual model (nM) predictions.

        One of 'allele' or 'alleles' must be specified. If 'allele' is specified
        all predictions will be for the given allele. If 'alleles' is specified
        it must be the same length as 'peptides' and give the allele
        corresponding to each peptide.

        Parameters
        ----------
        peptides : `EncodableSequences` or list of string
        alleles : list of string
        allele : string
        throw : boolean
            If True, a ValueError will be raised in the case of unsupported
            alleles or peptide lengths. If False, a warning will be logged and
            the predictions for the unsupported alleles or peptides will be NaN.
        centrality_measure : string or callable
            Measure of central tendency to use to combine predictions in the
            ensemble. Options include: mean, median, robust_mean.
        model_kwargs : dict
            Additional keyword arguments to pass to Class1NeuralNetwork.predict
        encoding_cache_dir : string, optional
            Directory for a shared BLOSUM62 peptide encoding cache. When set,
            the peptide encoding is computed once (or memmap-loaded if the
            cache entry already exists) and all networks in the ensemble hit
            the prepopulated cache instead of re-encoding. Useful when the
            same peptide list is scored by multiple predictors (e.g.
            comparing against a reference model) or across many calls.
            Uses ``self.neural_networks[0].hyperparameters["peptide_encoding"]``
            to determine the cache params. Silently falls back to direct
            encoding if the ensemble contains networks with heterogeneous
            peptide_encoding configs (none matches would cache-miss anyway).

        Returns
        -------
        numpy.array of predictions
        """
        df = self.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            allele=allele,
            throw=throw,
            include_percentile_ranks=False,
            include_confidence_intervals=False,
            centrality_measure=centrality_measure,
            model_kwargs=model_kwargs,
            encoding_cache_dir=encoding_cache_dir,
        )
        return df.prediction.values

    def predict_to_dataframe(
            self,
            peptides,
            alleles=None,
            allele=None,
            throw=True,
            include_individual_model_predictions=False,
            include_percentile_ranks=True,
            include_confidence_intervals=True,
            centrality_measure=DEFAULT_CENTRALITY_MEASURE,
            model_kwargs={},
            encoding_cache_dir=None):
        """
        Predict nM binding affinities. Gives more detailed output than `predict`
        method, including 5-95% prediction intervals.

        If multiple predictors are available for an allele, the predictions are
        the geometric means of the individual model predictions.

        One of 'allele' or 'alleles' must be specified. If 'allele' is specified
        all predictions will be for the given allele. If 'alleles' is specified
        it must be the same length as 'peptides' and give the allele
        corresponding to each peptide.

        Parameters
        ----------
        peptides : `EncodableSequences` or list of string
        alleles : list of string
        allele : string
        throw : boolean
            If True, a ValueError will be raised in the case of unsupported
            alleles or peptide lengths. If False, a warning will be logged and
            the predictions for the unsupported alleles or peptides will be NaN.
        include_individual_model_predictions : boolean
            If True, the predictions of each individual model are included as
            columns in the result DataFrame.
        include_percentile_ranks : boolean, default True
            If True, a "prediction_percentile" column will be included giving
            the percentile ranks. If no percentile rank info is available,
            this will be ignored with a warning.
        centrality_measure : string or callable
            Measure of central tendency to use to combine predictions in the
            ensemble. Options include: mean, median, robust_mean.
        model_kwargs : dict
            Additional keyword arguments to pass to Class1NeuralNetwork.predict

        Returns
        -------
        `pandas.DataFrame` of predictions
        """
        if isinstance(peptides, str):
            raise TypeError("peptides must be a list or array, not a string")
        if isinstance(alleles, str):
            raise TypeError("alleles must be a list or array, not a string")
        if allele is None and alleles is None:
            raise ValueError("Must specify 'allele' or 'alleles'.")

        peptides = self._prepare_peptides_with_optional_cache(
            peptides, encoding_cache_dir=encoding_cache_dir
        )
        df = pandas.DataFrame({
            'peptide': peptides.sequences
        }, copy=False)

        if allele is not None:
            if alleles is not None:
                raise ValueError("Specify exactly one of allele or alleles")
            normalized_allele = self.canonicalize_allele_name(allele)
            df["allele"] = normalized_allele
            df["normalized_allele"] = normalized_allele
            unique_alleles = [normalized_allele]
        else:
            df["allele"] = [
                self.canonicalize_allele_name(a) for a in alleles]
            df["normalized_allele"] = df["allele"]
            unique_alleles = df.normalized_allele.unique()

        if len(df) == 0:
            # No predictions.
            logging.warning("Predicting for 0 peptides.")
            empty_result = pandas.DataFrame(
                columns=[
                    'peptide',
                    'allele',
                    'prediction',
                    'prediction_low',
                    'prediction_high'
                ])
            return empty_result

        (min_peptide_length, max_peptide_length) = (
            self.supported_peptide_lengths)

        if (peptides.min_length < min_peptide_length or
                peptides.max_length > max_peptide_length):
            # Only compute this if needed
            all_peptides_supported = False
            sequence_length = df.peptide.str.len()
            df["supported_peptide"] = (
                (sequence_length >= min_peptide_length) &
                (sequence_length <= max_peptide_length))
            if (~df.supported_peptide).any():
                msg = (
                    "%d peptides have lengths outside of supported range [%d, %d]: "
                    "%s" % (
                        (~df.supported_peptide).sum(),
                        min_peptide_length,
                        max_peptide_length,
                        str(df.loc[~df.supported_peptide].peptide.unique())))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)
        else:
            # Handle common case efficiently.
            df["supported_peptide"] = True
            all_peptides_supported = True

        peptide_has_valid_amino_acids = (
            (~df.supported_peptide) |
            df.peptide.str.upper().str.match("^[ACDEFGHIKLMNPQRSTVWY]+$"))
        df["supported_peptide"] = (
                df["supported_peptide"] & peptide_has_valid_amino_acids)

        if (~peptide_has_valid_amino_acids).any():
            all_peptides_supported = False
            msg = (
                "%d peptides have nonstandard amino acids: "
                "%s" % (
                    (~peptide_has_valid_amino_acids).sum(),
                    str(df.loc[~peptide_has_valid_amino_acids].peptide.unique())))
            logging.warning(msg)
            if throw:
                raise ValueError(msg)

        num_pan_models = (
            len(self.class1_pan_allele_models)
            if not self.optimization_info.get("pan_models_merged", False)
            else self.optimization_info["num_pan_models_merged"])
        max_single_allele_models = max(
            len(self.allele_to_allele_specific_models.get(allele, []))
            for allele in unique_alleles
        )
        predictions_array = numpy.zeros(
            shape=(df.shape[0], num_pan_models + max_single_allele_models),
            dtype="float64")
        predictions_array[:] = numpy.nan

        if self.class1_pan_allele_models:
            master_allele_encoding = self.master_allele_encoding
            unsupported_alleles = [
                allele for allele in
                df.normalized_allele.unique()
                if allele not in self.allele_to_sequence
            ]
            if unsupported_alleles:
                truncate_at = 100
                allele_string = " ".join(
                    sorted(self.allele_to_sequence)[:truncate_at])
                if len(self.allele_to_sequence) > truncate_at:
                    allele_string += " + %d more alleles" % (
                        len(self.allele_to_sequence) - truncate_at)
                msg = (
                    "No sequences for allele(s): %s.\n"
                    "Supported alleles: %s" % (
                        " ".join(unsupported_alleles), allele_string))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)
            mask = df.supported_peptide & (
                ~df.normalized_allele.isin(unsupported_alleles))

            row_slice = None
            if mask is None or mask.all():
                row_slice = slice(None, None, None)  # all rows
                masked_allele_encoding = AlleleEncoding(
                    df.normalized_allele,
                    borrow_from=master_allele_encoding)
                masked_peptides = EncodableSequences.create(peptides)
            elif mask.sum() > 0:
                row_slice = mask
                masked_allele_encoding = AlleleEncoding(
                    df.loc[mask].normalized_allele,
                    borrow_from=master_allele_encoding)
                masked_peptides = EncodableSequences.create(
                    peptides.sequences[mask])

            if row_slice is not None:
                # The following line is a performance optimization that may be
                # revisited. It causes the neural network to set to include
                # only the alleles actually being predicted for. This makes
                # the network much smaller. However, subsequent calls to
                # predict will need to reset these weights, so there is a
                # tradeoff.
                masked_allele_encoding = masked_allele_encoding.compact()

                if self.optimization_info.get("pan_models_merged"):
                    # Multiple pan-allele models have been merged into one
                    # at the PyTorch level.
                    assert len(self.class1_pan_allele_models) == 1
                    predictions = self.class1_pan_allele_models[0].predict(
                        masked_peptides,
                        allele_encoding=masked_allele_encoding,
                        output_index=None,
                        **model_kwargs)
                    predictions_array[row_slice, :num_pan_models] = predictions
                else:
                    for (i, model) in enumerate(self.class1_pan_allele_models):
                        predictions_array[row_slice, i] = model.predict(
                            masked_peptides,
                            allele_encoding=masked_allele_encoding,
                            **model_kwargs)

        if self.allele_to_allele_specific_models:
            unsupported_alleles = [
                allele for allele in unique_alleles
                if not self.allele_to_allele_specific_models.get(allele)
            ]
            if unsupported_alleles:
                msg = (
                    "No single-allele models for allele(s): %s.\n"
                    "Supported alleles are: %s" % (
                        " ".join(unsupported_alleles),
                        " ".join(sorted(self.allele_to_allele_specific_models))))
                logging.warning(msg)
                if throw:
                    raise ValueError(msg)

            for allele in unique_alleles:
                models = self.allele_to_allele_specific_models.get(allele, [])
                if len(unique_alleles) == 1 and all_peptides_supported:
                    mask = None
                else:
                    mask = (
                        (df.normalized_allele == allele) &
                        df.supported_peptide).values

                row_slice = None
                if mask is None or mask.all():
                    peptides_for_allele = peptides
                    row_slice = slice(None, None, None)
                elif mask.sum() > 0:
                    peptides_for_allele = EncodableSequences.create(
                        df.loc[mask].peptide.values)
                    row_slice = mask

                if row_slice is not None:
                    for (i, model) in enumerate(models):
                        predictions_array[
                            row_slice,
                            num_pan_models + i,
                        ] = model.predict(peptides_for_allele, **model_kwargs)

        if callable(centrality_measure):
            centrality_function = centrality_measure
        else:
            centrality_function = CENTRALITY_MEASURES[centrality_measure]

        logs = numpy.log(predictions_array)
        row_has_predictions = (~numpy.isnan(logs)).any(axis=1)
        log_centers = numpy.full(df.shape[0], numpy.nan, dtype="float64")
        if row_has_predictions.any():
            log_centers[row_has_predictions] = centrality_function(
                logs[row_has_predictions]
            )
        df["prediction"] = numpy.exp(log_centers)

        if include_confidence_intervals:
            prediction_low = numpy.full(df.shape[0], numpy.nan, dtype="float64")
            prediction_high = numpy.full(df.shape[0], numpy.nan, dtype="float64")
            if row_has_predictions.any():
                prediction_low[row_has_predictions] = numpy.exp(
                    numpy.nanpercentile(logs[row_has_predictions], 5.0, axis=1)
                )
                prediction_high[row_has_predictions] = numpy.exp(
                    numpy.nanpercentile(logs[row_has_predictions], 95.0, axis=1)
                )
            df["prediction_low"] = prediction_low
            df["prediction_high"] = prediction_high

        if include_individual_model_predictions:
            for i in range(num_pan_models):
                df["model_pan_%d" % i] = predictions_array[:, i]

            for i in range(max_single_allele_models):
                df["model_single_%d" % i] = predictions_array[
                    :, num_pan_models + i
                ]

        if include_percentile_ranks:
            if self.allele_to_percent_rank_transform:
                df["prediction_percentile"] = self.percentile_ranks(
                    df.prediction,
                    alleles=df.normalized_allele.values,
                    throw=throw)
            else:
                warnings.warn("No percentile rank information available.")

        del df["supported_peptide"]
        del df["normalized_allele"]
        return df

    def calibrate_percentile_ranks(
            self,
            peptides=None,
            num_peptides_per_length=int(1e5),
            alleles=None,
            bins=None,
            motif_summary=False,
            summary_top_peptide_fractions=[0.001],
            verbose=False,
            model_kwargs={}):
        """
        Compute the cumulative distribution of ic50 values for a set of alleles
        over a large universe of random peptides, to enable taking quantiles
        of this distribution later.

        Parameters
        ----------
        peptides : sequence of string or EncodableSequences, optional
            Peptides to use
        num_peptides_per_length : int, optional
            If peptides argument is not specified, then num_peptides_per_length
            peptides are randomly sampled from a uniform distribution for each
            supported length
        alleles : sequence of string, optional
            Alleles to perform calibration for. If not specified all supported
            alleles will be calibrated.
        bins : object
            Anything that can be passed to numpy.histogram's "bins" argument
            can be used here, i.e. either an integer or a sequence giving bin
            edges. This is in ic50 space.
        motif_summary : bool
            If True, the length distribution and per-position amino acid
            frequencies are also calculated for the top x fraction of tightest-
            binding peptides, where each value of x is given in the
            summary_top_peptide_fractions list.
        summary_top_peptide_fractions : list of float
            Only used if motif_summary is True
        verbose : boolean
            Whether to print status updates to stdout
        model_kwargs : dict
            Additional low-level Class1NeuralNetwork.predict() kwargs.

        Returns
        ----------
        dict of string -> pandas.DataFrame

        If motif_summary is True, this will have keys  "frequency_matrices" and
        "length_distributions". Otherwise it will be empty.

        """
        if bins is None:
            bins = to_ic50(numpy.linspace(1, 0, 1000))

        if alleles is None:
            alleles = self.supported_alleles

        if peptides is None:
            peptides = []
            lengths = range(
                self.supported_peptide_lengths[0],
                self.supported_peptide_lengths[1] + 1)
            for length in lengths:
                peptides.extend(
                    random_peptides(num_peptides_per_length, length))

        encoded_peptides = EncodableSequences.create(peptides)

        if motif_summary:
            frequency_matrices = []
            length_distributions = []
        else:
            frequency_matrices = None
            length_distributions = None
        for allele in alleles:
            start = time.time()
            predictions = self.predict(
                encoded_peptides, allele=allele, model_kwargs=model_kwargs)
            if verbose:
                elapsed = time.time() - start
                print(
                    "Generated %d predictions for allele %s in %0.2f sec: "
                    "%0.2f predictions / sec" % (
                        len(encoded_peptides.sequences),
                        allele,
                        elapsed,
                        len(encoded_peptides.sequences) / elapsed))
            transform = PercentRankTransform()
            transform.fit(predictions, bins=bins)
            self.allele_to_percent_rank_transform[allele] = transform

            if frequency_matrices is not None:
                predictions_df = pandas.DataFrame({
                    'peptide': encoded_peptides.sequences,
                    'prediction': predictions
                }).drop_duplicates('peptide').set_index("peptide")
                predictions_df["length"] = predictions_df.index.str.len()
                for (length, sub_df) in predictions_df.groupby("length"):
                    for cutoff_fraction in summary_top_peptide_fractions:
                        selected = sub_df.prediction.nsmallest(
                            max(
                                int(len(sub_df) * cutoff_fraction),
                                1)).index.values
                        matrix = positional_frequency_matrix(selected).reset_index()
                        original_columns = list(matrix.columns)
                        matrix["allele"] = allele
                        matrix["length"] = length
                        matrix["cutoff_fraction"] = cutoff_fraction
                        matrix["cutoff_count"] = len(selected)
                        matrix = matrix[
                            ["allele", "length", "cutoff_fraction", "cutoff_count"]
                            + original_columns
                        ]
                        frequency_matrices.append(matrix)

                # Length distribution
                for cutoff_fraction in summary_top_peptide_fractions:
                    cutoff_count = max(
                        int(len(predictions_df) * cutoff_fraction), 1)
                    length_distribution = predictions_df.prediction.nsmallest(
                        cutoff_count).index.str.len().value_counts()
                    length_distribution.index.name = "length"
                    length_distribution /= length_distribution.sum()
                    length_distribution = length_distribution.to_frame()
                    length_distribution.columns = ["fraction"]
                    length_distribution = length_distribution.reset_index()
                    length_distribution["allele"] = allele
                    length_distribution["cutoff_fraction"] = cutoff_fraction
                    length_distribution["cutoff_count"] = cutoff_count
                    length_distribution = length_distribution[[
                        "allele",
                        "cutoff_fraction",
                        "cutoff_count",
                        "length",
                        "fraction"
                    ]].sort_values(["cutoff_fraction", "length"])
                    length_distributions.append(length_distribution)

        if frequency_matrices is not None:
            frequency_matrices = pandas.concat(
                frequency_matrices, ignore_index=True)

        if length_distributions is not None:
            length_distributions = pandas.concat(
                length_distributions, ignore_index=True)

        if motif_summary:
            return {
                'frequency_matrices': frequency_matrices,
                'length_distributions': length_distributions,
            }
        return {}

    @staticmethod
    def _auto_size_calibration_batches(
            model, device, n_peptides, n_alleles,
            num_workers_per_gpu=1,
            free_memory_fraction=0.4,
            num_cached_networks=1,
            peptide_stage_dim=None):
        """Split the auto-sized batch budget between peptide and allele axes.

        Calibrate's peak VRAM is dominated by ``cached_stages`` — all
        ``num_cached_networks`` networks' peptide-stage outputs held on
        device simultaneously, sized ``n_peptides × peptide_stage_dim ×
        4 bytes`` per network. That footprint is fixed (independent of
        batch size) and must be subtracted before sizing the
        peptide/allele forward batches, otherwise auto-sizing gives a
        budget that fits a single forward but not the forward + cache.
        """
        from .class1_neural_network import (
            compute_prediction_batch_size,
            _free_device_memory_bytes,
            _estimate_peak_bytes_per_row,
            _AUTO_BATCH_MAX_ROWS,
            _AUTO_BATCH_MIN_ROWS,
        )
        if n_peptides == 0 or n_alleles == 0:
            return max(n_peptides, 1), max(n_alleles, 1)
        if device.type != "cuda":
            total_rows = compute_prediction_batch_size(
                device,
                model=model,
                num_workers_per_gpu=num_workers_per_gpu,
                free_memory_fraction=free_memory_fraction,
            )
        else:
            workers = max(int(num_workers_per_gpu), 1)
            free = _free_device_memory_bytes(device)
            peak_bytes = _estimate_peak_bytes_per_row(model)
            per_worker_budget = int(free * float(free_memory_fraction) / workers)
            stage_dim = peptide_stage_dim
            if stage_dim is None:
                # MergedClass1NeuralNetwork: peptide-stage cache is the
                # concatenation of all sub-networks' stages → sum the
                # per-sub-network stage dims. Otherwise fall back to a
                # conservative encoding-shape probe.
                sub_networks = getattr(model, "networks", None)
                if (
                    sub_networks is not None
                    and not hasattr(model, "peptide_encoding_shape")
                ):
                    try:
                        sub_dims = []
                        for net in sub_networks:
                            sub_enc = getattr(net, "peptide_encoding_shape", None)
                            if sub_enc is not None:
                                sub_dims.append(int(sub_enc[0]) * int(sub_enc[1]))
                            else:
                                sub_dims.append(1024)
                        stage_dim = int(sum(sub_dims))
                    except Exception:
                        stage_dim = None
                if stage_dim is None:
                    try:
                        enc_shape = getattr(model, "peptide_encoding_shape", None)
                        if enc_shape is not None:
                            stage_dim = int(enc_shape[0]) * int(enc_shape[1])
                    except Exception:
                        stage_dim = None
                if stage_dim is None:
                    stage_dim = peak_bytes // 32 if peak_bytes else 1024
            cache_bytes = (
                int(num_cached_networks)
                * int(n_peptides)
                * int(stage_dim)
                * 4
            )
            forward_budget = per_worker_budget - cache_bytes
            if forward_budget < peak_bytes * _AUTO_BATCH_MIN_ROWS:
                logging.warning(
                    "calibrate auto-sizer: cached_stages footprint "
                    "(%.2f GB) exceeds %.0f%% of per-worker budget "
                    "(%.2f GB free / %d workers). Falling back to "
                    "minimum batch; consider --max-workers-per-gpu 1.",
                    cache_bytes / 1e9,
                    free_memory_fraction * 100,
                    free / 1e9,
                    workers,
                )
                forward_budget = peak_bytes * _AUTO_BATCH_MIN_ROWS
            total_rows = max(
                _AUTO_BATCH_MIN_ROWS,
                min(forward_budget // peak_bytes, _AUTO_BATCH_MAX_ROWS),
            )
        peptide_batch = min(n_peptides, max(2_000, total_rows // 64))
        allele_batch = min(n_alleles, max(1, total_rows // peptide_batch), 256)
        while allele_batch * peptide_batch > total_rows and peptide_batch > 2_000:
            peptide_batch = max(2_000, peptide_batch // 2)
        return int(peptide_batch), int(allele_batch)

    @staticmethod
    def _prepare_motif_summary_state_gpu(encoded_peptides, device):
        """One-time setup for GPU motif_summary; lifts the per-allele
        ``drop_duplicates`` + length-bucket + AA-encoding work out of
        the calibrate chunk loop so each chunk is pure tensor math.

        Returns a dict of device-resident tensors:
            unique_idx_t : (n_unique,) long — first-occurrence indices
                into the full peptide list. Selecting columns with this
                from the chunk's ``ic50_device`` reproduces the legacy
                ``drop_duplicates('peptide')`` semantics (first row wins).
            length_groups : dict[L] -> (n_at_L,) long indices into the
                unique-peptide axis for peptides of length L.
            aa_codes_per_length : dict[L] -> (n_at_L, L) long tensor of
                amino-acid index codes (matches ``AMINO_ACID_INDEX``,
                so X = 20 if it ever appears).
            unique_lengths_t : (n_unique,) long — peptide length per
                unique peptide; powers the per-allele length distribution.
            n_unique : int.
        """
        import torch
        from .amino_acid import AMINO_ACID_INDEX

        seqs = list(encoded_peptides.sequences)
        seen = set()
        unique_idx = []
        for i, s in enumerate(seqs):
            if s not in seen:
                seen.add(s)
                unique_idx.append(i)
        unique_idx = numpy.asarray(unique_idx, dtype=numpy.int64)
        unique_seqs = [seqs[i] for i in unique_idx]
        lengths = numpy.fromiter(
            (len(p) for p in unique_seqs),
            dtype=numpy.int64,
            count=len(unique_seqs),
        )

        unique_lengths_t = torch.from_numpy(lengths).to(device)
        unique_idx_t = torch.from_numpy(unique_idx).to(device)

        # ASCII -> AA-index lookup so the per-residue encoding is one
        # vectorized numpy gather instead of 8M Python ops over 800k
        # peptides. Unknown bytes default to 0 ("A"), matching the
        # implicit "trust caller" contract elsewhere in mhcflurry's
        # peptide handling — random_peptides only emits the 20 common
        # AAs so this is unreachable in the calibrate path.
        ascii_lut = numpy.zeros(256, dtype=numpy.int64)
        for letter, idx in AMINO_ACID_INDEX.items():
            if len(letter) == 1:
                ascii_lut[ord(letter)] = idx

        length_groups = {}
        aa_codes_per_length = {}
        for L_np in numpy.unique(lengths):
            L = int(L_np)
            sel = numpy.where(lengths == L)[0].astype(numpy.int64)
            n_at_L = int(sel.shape[0])
            joined = "".join(unique_seqs[i] for i in sel).encode("ascii")
            byte_arr = numpy.frombuffer(
                joined, dtype=numpy.uint8,
            ).reshape(n_at_L, L)
            codes = ascii_lut[byte_arr]  # (n_at_L, L) int64
            length_groups[L] = torch.from_numpy(sel).to(device)
            aa_codes_per_length[L] = torch.from_numpy(
                numpy.ascontiguousarray(codes),
            ).to(device)

        return {
            "unique_idx_t": unique_idx_t,
            "length_groups": length_groups,
            "aa_codes_per_length": aa_codes_per_length,
            "unique_lengths_t": unique_lengths_t,
            "n_unique": int(unique_idx.shape[0]),
        }

    @staticmethod
    def _motif_summary_chunk_gpu(
            ic50_device,
            state,
            summary_top_peptide_fractions,
            batch_alleles):
        """GPU motif_summary for one allele chunk.

        Replaces the per-allele
        ``DataFrame(...).drop_duplicates().groupby('length').nsmallest(...)``
        block in the fast path with vectorized tensor ops:

        * ``torch.topk(largest=False)`` selects top-k tightest binders
          per (allele, length) — kernel-tuned partial sort across all
          alleles in the chunk in one launch instead of 30× per-allele
          pandas ``nsmallest`` calls.
        * AA frequency matrices are computed by gathering precomputed
          per-length AA-code tensors and ``scatter_add`` into a one-hot
          counts buffer of shape ``(a_size, L, 20)``; pandas only
          assembles the final per-row schema.
        * Length distributions are ``topk`` over the full
          unique-peptide axis followed by a per-row ``scatter_add``
          (a row-wise ``bincount``).

        Returns ``(freq_matrix_dfs, length_dist_dfs)`` — lists of
        ``pandas.DataFrame`` matching the legacy slow path's per-row
        schema, ready to ``pandas.concat`` once all chunks are done.
        """
        import torch
        from .amino_acid import AMINO_ACIDS

        a_size = len(batch_alleles)
        device = ic50_device.device
        n_unique = state["n_unique"]
        # AMINO_ACIDS = COMMON_AMINO_ACIDS_WITH_UNKNOWN keys, alphabetical
        # then X — first 20 entries are the BLOSUM62-ordered non-X rows
        # the legacy ``positional_frequency_matrix`` returns.
        aa_columns = AMINO_ACIDS[:20]

        ic50_unique = ic50_device.index_select(1, state["unique_idx_t"])

        freq_matrices = []
        length_dists = []

        for cutoff_fraction in summary_top_peptide_fractions:
            for L, idx_in_unique in state["length_groups"].items():
                n_at_L = int(idx_in_unique.numel())
                k = max(int(n_at_L * cutoff_fraction), 1)
                ic50_L = ic50_unique.index_select(1, idx_in_unique)
                top_idx = torch.topk(
                    ic50_L, k, dim=1, largest=False, sorted=False,
                ).indices  # (a_size, k)
                codes = state["aa_codes_per_length"][L]  # (n_at_L, L) long
                selected_codes = codes[top_idx]  # (a_size, k, L) long
                # Permute to (a_size, L, k) so scatter_add packs counts
                # along the last (AA) axis. We allocate 21 AA slots to
                # absorb X (index 20) and discard the X column at the
                # end — this matches the legacy semantics where
                # ``positional_frequency_matrix``'s row index excludes X
                # while the divisor stays at ``k`` (so positions with
                # any X residues sum to <1 across the 20 columns).
                scatter_dst = selected_codes.permute(0, 2, 1)
                counts = torch.zeros(
                    a_size, L, 21, dtype=torch.float64, device=device,
                )
                counts.scatter_add_(
                    2, scatter_dst,
                    torch.ones_like(scatter_dst, dtype=torch.float64),
                )
                freq_cpu = (counts[:, :, :20] / float(k)).cpu().numpy()
                for ai, allele in enumerate(batch_alleles):
                    df = pandas.DataFrame(
                        freq_cpu[ai],
                        index=numpy.arange(1, L + 1),
                        columns=aa_columns,
                    )
                    df.index.name = "position"
                    df = df.reset_index()
                    df.insert(0, "allele", allele)
                    df.insert(1, "length", L)
                    df.insert(2, "cutoff_fraction", cutoff_fraction)
                    df.insert(3, "cutoff_count", k)
                    freq_matrices.append(df)

            k_total = max(int(n_unique * cutoff_fraction), 1)
            top_full_idx = torch.topk(
                ic50_unique, k_total, dim=1, largest=False, sorted=False,
            ).indices  # (a_size, k_total)
            lengths_per_topk = state["unique_lengths_t"][top_full_idx]
            max_len_p1 = int(state["unique_lengths_t"].max().item()) + 1
            length_counts = torch.zeros(
                a_size, max_len_p1, dtype=torch.float64, device=device,
            )
            length_counts.scatter_add_(
                1, lengths_per_topk,
                torch.ones_like(lengths_per_topk, dtype=torch.float64),
            )
            length_fractions_cpu = (
                length_counts / float(k_total)
            ).cpu().numpy()
            for ai, allele in enumerate(batch_alleles):
                row = length_fractions_cpu[ai]
                present = numpy.where(row > 0)[0]
                if present.size == 0:
                    continue
                ld = pandas.DataFrame({
                    "allele": allele,
                    "cutoff_fraction": cutoff_fraction,
                    "cutoff_count": k_total,
                    "length": present.astype(numpy.int64),
                    "fraction": row[present],
                })[[
                    "allele", "cutoff_fraction", "cutoff_count",
                    "length", "fraction",
                ]].sort_values(
                    ["cutoff_fraction", "length"]
                ).reset_index(drop=True)
                length_dists.append(ld)

        return freq_matrices, length_dists

    def calibrate_percentile_ranks_fast(
            self,
            peptides,
            alleles,
            bins=None,
            motif_summary=False,
            summary_top_peptide_fractions=(0.001,),
            allele_batch_size="auto",
            peptide_batch_size="auto",
            num_workers_per_gpu=1,
            device=None,
            verbose=False):
        """GPU-hoisted calibration for many alleles sharing a peptide set.

        Drop-in replacement for the bulk of ``calibrate_percentile_ranks``
        when calibration is dominated by per-allele Python dispatch (the
        pan-allele full-universe calibration workload, which sweeps
        ~20k alleles across the same peptide set). Two structural changes:

        1. **Precompute peptide-stage activations once per network.** The
           network's locally-connected + peptide-dense + early-batchnorm
           layers are identity across alleles; we forward the calibration
           peptides through that stage exactly once per network and
           cache the output tensor on device.
        2. **Batch ``allele_batch_size`` alleles per forward.** A single
           ``forward_from_peptide_stage`` call replaces
           ``allele_batch_size`` separate ``model.predict`` calls —
           amortizing all the small kernel-launch + Python-dispatch
           overhead that was dominating wall-clock time on the pan-allele
           full-universe calibration. Effective batch size per forward:
           ``allele_batch_size * peptide_batch_size``, sized so it fits
           comfortably in an A100's 80 GB.

        Semantics-preserving w.r.t. ``calibrate_percentile_ranks``: same
        peptides → same per-network IC50 predictions (bit-identical when
        the network is deterministic) → same geometric-mean ensemble
        aggregation → same ``PercentRankTransform.fit`` per allele.
        Only the schedule of Python dispatch and GPU kernel launches
        changes.

        Only handles pan-allele models. Mass-spec-only models and
        ``class1_presentation_predictor`` should keep using the slower
        per-allele path.

        Parameters
        ----------
        peptides : sequence of string or EncodableSequences
        alleles : sequence of string — already-normalized allele names
            (canonicalization is the caller's responsibility for speed).
        bins : argument to ``numpy.histogram`` (default: 999 bin edges
            in IC50 space, matching ``calibrate_percentile_ranks``).
        motif_summary : bool — populate frequency matrices + length
            distributions, same format as ``calibrate_percentile_ranks``.
        summary_top_peptide_fractions : iterable of float — only used
            when ``motif_summary=True``.
        allele_batch_size : int — how many alleles share a single forward
            through the merge + main dense. 64 is a reasonable default
            on an A100-80GB with the release pan-allele arch + the 800k
            peptide calibration set (peak VRAM ~ allele_batch_size *
            peptide_batch_size * 4 bytes per hidden unit).
        peptide_batch_size : int — peptide chunk size on device.
        device : str or torch.device — defaults to CUDA if available.
        verbose : bool — per-batch timing to stdout.

        Returns
        -------
        dict (same schema as ``calibrate_percentile_ranks``). Also
        populates ``self.allele_to_percent_rank_transform[allele]`` for
        every allele in ``alleles``.
        """
        import torch

        from .class1_neural_network import Class1NeuralNetwork  # noqa
        from .encodable_sequences import EncodableSequences
        from .allele_encoding import AlleleEncoding
        from .regression_target import to_ic50
        from .percent_rank_transform import PercentRankTransform

        if not self.class1_pan_allele_models:
            raise ValueError(
                "calibrate_percentile_ranks_fast is pan-allele-only; "
                "this predictor has no pan-allele models."
            )

        if bins is None:
            bins = to_ic50(numpy.linspace(1, 0, 1000))
        # numpy.histogram accepts bins as either an int (n_bins) or an
        # array of edges. The fast path requires explicit edges so we
        # can bucketize on device. Materialize ints into edges here.
        bin_edges_array = numpy.asarray(bins)
        if bin_edges_array.ndim == 0:
            n_bins_int = int(bin_edges_array)
            assert n_bins_int > 0
            bin_edges_array = to_ic50(
                numpy.linspace(1, 0, n_bins_int + 1)
            )

        if device is None:
            device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = torch.device(device)

        encoded_peptides = EncodableSequences.create(peptides)
        n_peptides = len(encoded_peptides.sequences)
        if n_peptides == 0:
            raise ValueError("No peptides supplied for calibration")
        alleles = list(alleles)
        if not alleles:
            raise ValueError("No alleles supplied for calibration")

        master = self.master_allele_encoding
        unknown = [a for a in alleles if a not in master.allele_to_index]
        if unknown:
            raise KeyError(
                "calibrate_percentile_ranks_fast: %d allele(s) not in the "
                "predictor's master encoding (first 5: %s)" % (
                    len(unknown), unknown[:5],
                )
            )
        allele_idx_np = numpy.array(
            [master.allele_to_index[a] for a in alleles], dtype=numpy.int64,
        )

        # Per-network prep: ensure allele representations are wired up,
        # move the eager (uncompiled) model to device, cache the
        # peptide-stage output.
        networks = self.class1_pan_allele_models
        if allele_batch_size in (None, "auto") or peptide_batch_size in (None, "auto"):
            # Resolve once, using the first network as the architecture
            # probe — all ensembles we serve have homogeneous layer
            # sizing, so one probe is sufficient.
            probe_net = networks[0].network(borrow=True)
            probe_net.to(device)
            auto_peptide, auto_allele = self._auto_size_calibration_batches(
                probe_net, device, n_peptides, len(alleles),
                num_workers_per_gpu=num_workers_per_gpu,
                num_cached_networks=len(networks),
            )
            if peptide_batch_size in (None, "auto"):
                peptide_batch_size = auto_peptide
            if allele_batch_size in (None, "auto"):
                allele_batch_size = auto_allele
            if verbose:
                print(
                    "calibrate_percentile_ranks_fast auto-sized: "
                    f"peptide_batch={peptide_batch_size}, "
                    f"allele_batch={allele_batch_size} "
                    f"(workers_per_gpu={num_workers_per_gpu}, "
                    f"cached_networks={len(networks)})"
                )
        peptide_batch_size = int(peptide_batch_size)
        allele_batch_size = int(allele_batch_size)
        cached_stages = []  # list of (net_obj, model, peptide_stage on device)
        for net_obj in networks:
            (_, allele_reps) = net_obj.allele_encoding_to_network_input(
                AlleleEncoding(alleles=[], borrow_from=master),
            )
            net_obj.set_allele_representations(allele_reps)
            model = net_obj.network(borrow=True)
            model.to(device)
            model.eval()
            peptide_input = net_obj.peptides_to_network_input(encoded_peptides)
            peptide_is_indices = net_obj.uses_peptide_torch_encoding()
            stage_parts = []
            with torch.no_grad():
                for start in range(0, n_peptides, peptide_batch_size):
                    end = min(start + peptide_batch_size, n_peptides)
                    chunk = peptide_input[start:end]
                    keep_int = (
                        peptide_is_indices
                        and chunk.ndim == 2
                        and numpy.issubdtype(chunk.dtype, numpy.integer)
                    )
                    if keep_int:
                        tensor = torch.from_numpy(chunk).to(device)
                    else:
                        tensor = torch.from_numpy(
                            numpy.asarray(chunk, dtype=numpy.float32),
                        ).to(device)
                    stage_parts.append(model.forward_peptide_stage(tensor))
            cached_stages.append((net_obj, model, torch.cat(stage_parts, dim=0)))
            del peptide_input

        log50000 = float(numpy.log(50000.0))
        n_alleles = len(alleles)
        frequency_matrices = [] if motif_summary else None
        length_distributions = [] if motif_summary else None
        # Bin edges for the on-device batched histogram fit. Materialize
        # once outside the per-chunk loop — same edges across alleles.
        bins_tensor = torch.as_tensor(
            bin_edges_array, dtype=torch.float64, device=device,
        )

        # Hoist the per-allele dedup + length-bucket + AA-encoding work
        # out of the chunk loop. The peptide universe is identical across
        # alleles so this only needs to run once per calibrate call.
        motif_state = (
            self._prepare_motif_summary_state_gpu(encoded_peptides, device)
            if motif_summary else None
        )

        for abatch_start in range(0, n_alleles, allele_batch_size):
            abatch_end = min(abatch_start + allele_batch_size, n_alleles)
            a_size = abatch_end - abatch_start
            batch_alleles = alleles[abatch_start:abatch_end]
            batch_idx = torch.from_numpy(
                allele_idx_np[abatch_start:abatch_end],
            ).to(device)

            # log-space accumulator across networks: (a_size, n_peptides).
            # Same aggregation scheme as the existing ensemble code:
            # arithmetic mean of per-network log(IC50) = geometric mean
            # of per-network IC50. Use fp64 where supported so the
            # aggregation matches the legacy path bit-for-bit; fall
            # back to fp32 on MPS (which has no fp64) — drift there is
            # ~1e-6 in log-IC50, well below histogram-bin resolution.
            accum_dtype = (
                torch.float32 if device.type == "mps" else torch.float64
            )
            log_ic50_sum = torch.zeros(
                a_size, n_peptides, dtype=accum_dtype, device=device,
            )

            for (_, model, peptide_stage) in cached_stages:
                with torch.no_grad():
                    for start in range(0, n_peptides, peptide_batch_size):
                        end = min(start + peptide_batch_size, n_peptides)
                        p_chunk = peptide_stage[start:end]  # (chunk_n, d)
                        network_output = (
                            model.forward_cartesian_from_peptide_stage(
                                p_chunk,
                                batch_idx,
                            )
                        )
                        # shape (a_size, chunk_n, num_outputs) — take the first
                        # output (matches existing predict default).
                        network_output = network_output[..., 0]
                        # log(IC50) = (1 - network_output) * log(50000).
                        # Network output ∈ [0,1] from sigmoid — never
                        # negative, so the log(0) edge is avoided without
                        # clamping.
                        log_ic50_sum[:, start:end] += (
                            (1.0 - network_output).to(accum_dtype) * log50000
                        )

            log_mean = log_ic50_sum / float(len(cached_stages))
            ic50_device = torch.exp(log_mean)  # (a_size, n_peptides) on device
            # Batched torch fit replaces the per-allele numpy.histogram
            # loop that dominated calibrate wall time. One bucketize +
            # one scatter_add covers all 30 alleles in the chunk; the
            # final per-allele cdf is materialized as numpy only at
            # storage time so the persistent format is unchanged.
            transforms = PercentRankTransform.fit_batch_torch(
                ic50_device, bins_tensor,
            )
            for local_i, allele in enumerate(batch_alleles):
                self.allele_to_percent_rank_transform[allele] = transforms[local_i]
            if motif_summary:
                # All motif-summary work now runs on device — topk +
                # scatter_add for AA frequency matrices, topk + scatter_add
                # bincount for length distributions. Pandas only assembles
                # final per-row schema at chunk-end. This replaces the
                # per-allele DataFrame.drop_duplicates().groupby().nsmallest()
                # block that dominated calibrate wall time after the
                # PercentRankTransform.fit GPU port.
                chunk_freq, chunk_ld = self._motif_summary_chunk_gpu(
                    ic50_device,
                    motif_state,
                    summary_top_peptide_fractions,
                    batch_alleles,
                )
                frequency_matrices.extend(chunk_freq)
                length_distributions.extend(chunk_ld)
            if verbose:
                print(
                    "calibrate_percentile_ranks_fast: "
                    f"{abatch_end}/{n_alleles} alleles done"
                )

        if motif_summary:
            return {
                "frequency_matrices": pandas.concat(
                    frequency_matrices, ignore_index=True,
                ),
                "length_distributions": pandas.concat(
                    length_distributions, ignore_index=True,
                ),
            }
        return {}

    def model_select(
            self,
            score_function,
            alleles=None,
            min_models=1,
            max_models=10000):
        """
        Perform model selection using a user-specified scoring function.

        This works only with allele-specific models, not pan-allele models.

        Model selection is done using a "step up" variable selection procedure,
        in which models are repeatedly added to an ensemble until the score
        stops improving.

        Parameters
        ----------
        score_function : Class1AffinityPredictor -> float function
            Scoring function

        alleles : list of string, optional
            If not specified, model selection is performed for all alleles.

        min_models : int, optional
            Min models to select per allele

        max_models : int, optional
            Max models to select per allele

        Returns
        -------
        Class1AffinityPredictor : predictor containing the selected models
        """

        if alleles is None:
            alleles = self.supported_alleles

        dfs = []
        allele_to_allele_specific_models = {}
        for allele in alleles:
            df = pandas.DataFrame({
                'model': self.allele_to_allele_specific_models[allele]
            })
            df["model_num"] = df.index
            df["allele"] = allele
            df["selected"] = False

            round_num = 1

            while not df.selected.all() and sum(df.selected) < max_models:
                score_col = "score_%2d" % round_num
                prev_score_col = "score_%2d" % (round_num - 1)

                existing_selected = list(df[df.selected].model)
                df[score_col] = [
                    numpy.nan if row.selected else
                    score_function(
                        Class1AffinityPredictor(
                            allele_to_allele_specific_models={
                                allele: [row.model] + existing_selected
                            }
                        )
                    )
                    for (_, row) in df.iterrows()
                ]

                if round_num > min_models and (
                        df[score_col].max() < df[prev_score_col].max()):
                    break

                # In case of a tie, pick a model at random.
                (best_model_index,) = df.loc[
                    (df[score_col] == df[score_col].max())
                ].sample(1).index
                df.loc[best_model_index, "selected"] = True
                round_num += 1

            dfs.append(df)
            allele_to_allele_specific_models[allele] = list(
                df.loc[df.selected].model)

        df = pandas.concat(dfs, ignore_index=True)

        new_predictor = Class1AffinityPredictor(
            allele_to_allele_specific_models,
            metadata_dataframes={
                "model_selection": df,
            })
        return new_predictor
