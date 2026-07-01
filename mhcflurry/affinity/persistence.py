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

"""Persistence helpers for class I affinity predictors."""

import collections
import json
import logging
import time
from functools import partial
from getpass import getuser
from os import mkdir
from os.path import abspath, exists, join
from socket import gethostname

import numpy
import pandas

from ..class1_neural_network import Class1NeuralNetwork
from ..common import load_weights, normalize_allele_name, save_weights
from ..downloads import get_default_class1_models_dir
from ..percent_rank_transform import PercentRankTransform
from ..pseudosequences import (
    LEGACY_ALLELE_SEQUENCES_FILENAME,
    pseudosequence_filename_candidates,
    pseudosequence_filename_for_mapping,
)
from ..version import __version__


def save_predictor(predictor, models_dir, model_names_to_write=None, write_metadata=True):
    """
    Serialize the predictor to a directory on disk. If the directory does
    not exist it will be created.

    The serialization format consists of a file called "manifest.csv" with
    the configurations of each Class1NeuralNetwork, along with per-network
    files giving the model weights. If there are pan-allele predictors in
    the ensemble, the pseudosequences are also stored in the
    directory. There is also a small file "info.txt" with basic metadata:
    when the models were trained, by whom, on what host.

    Parameters
    ----------
    models_dir : string
        Path to directory. It will be created if it doesn't exist.

    model_names_to_write : list of string, optional
        Only write the weights for the specified models. Useful for
        incremental updates during training. Passing an explicit empty
        list writes no model artifacts; this is used by calibration-only
        updates that should replace ``percent_ranks.csv`` without touching
        the manifest, weights, model provenance, allele sequences, or
        optimization metadata. Explicit ``metadata_dataframes`` are still
        written when ``write_metadata`` is true.

    write_metadata : boolean, optional
        Whether to write optional metadata
    """
    if model_names_to_write is None:
        # Write all models
        model_names_to_write = list(predictor.manifest_df.model_name.values)
        write_model_artifacts = True
    else:
        model_names_to_write = list(model_names_to_write)
        write_model_artifacts = len(model_names_to_write) > 0

    if write_model_artifacts:
        predictor.check_consistency()

    if not exists(models_dir):
        mkdir(models_dir)

    if write_model_artifacts:
        sub_manifest_df = predictor.manifest_df.loc[
            predictor.manifest_df.model_name.isin(model_names_to_write)
        ].copy()

        # Network JSON configs may have changed since the models were added,
        # for example due to changes to the allele representation layer.
        # So we update the JSON configs here also.
        updated_network_config_jsons = []
        for (_, row) in sub_manifest_df.iterrows():
            updated_network_config_jsons.append(
                json.dumps(row.model.get_config()))
            weights_path = predictor.weights_path(models_dir, row.model_name)
            save_weights(row.model.get_weights(), weights_path)
            logging.info("Wrote: %s", weights_path)
        sub_manifest_df["config_json"] = updated_network_config_jsons
        predictor.manifest_df.loc[
            sub_manifest_df.index,
            "config_json"
        ] = updated_network_config_jsons

        write_manifest_df = predictor.manifest_df[[
            c for c in predictor.manifest_df.columns if c != "model"
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

        # Save pseudosequences. New artifacts get an explicit
        # pseudosequences.<source>.<length>aa.csv filename while still
        # writing allele_sequences.csv for older MHCflurry releases and
        # external tooling.
        if predictor.allele_to_sequence is not None:
            allele_to_sequence_df = pandas.DataFrame(
                list(predictor.allele_to_sequence.items()),
                columns=['allele', 'sequence']
            )
            legacy_path = join(models_dir, LEGACY_ALLELE_SEQUENCES_FILENAME)
            allele_to_sequence_df.to_csv(legacy_path, index=False)
            logging.info("Wrote: %s", legacy_path)

            pseudosequences_filename = (
                pseudosequence_filename_for_mapping(
                    predictor.allele_to_sequence))
            if pseudosequences_filename is not None:
                pseudosequences_path = join(
                    models_dir, pseudosequences_filename)
                pseudosequences_df = pandas.DataFrame(
                    list(predictor.allele_to_sequence.items()),
                    columns=['allele', 'pseudosequence'])
                pseudosequences_df.to_csv(
                    pseudosequences_path, index=False)
                logging.info("Wrote: %s", pseudosequences_path)
            else:
                logging.info(
                    "Pseudosequences have mixed or unknown lengths; "
                    "skipping explicit pseudosequences.*.*aa.csv alias.")

    if write_metadata and predictor.metadata_dataframes:
        for (name, df) in predictor.metadata_dataframes.items():
            metadata_df_path = join(models_dir, "%s.csv.bz2" % name)
            df.to_csv(metadata_df_path, index=False, compression="bz2")

    if predictor.allele_to_percent_rank_transform:
        percent_ranks_df = None
        for (allele, transform) in predictor.allele_to_percent_rank_transform.items():
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

    if write_model_artifacts and predictor.optimization_info:
        # If the model being saved was optimized, we need to save that
        # information since it can affect how predictions are performed
        # (e.g. stitched-together ensembles output concatenated results,
        # which then need to be averaged outside the model).
        optimization_info_path = join(models_dir, "optimization_info.json")
        with open(optimization_info_path, "w") as fd:
            json.dump(predictor.optimization_info, fd, indent=4)
    predictor.models_dir = abspath(models_dir)


def load_predictor(
        predictor_class,
        models_dir=None,
        max_models=None,
        optimization_level=None,
        optimization_level_default=None):
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
        optimization_level = optimization_level_default

    manifest_path = join(models_dir, "manifest.csv")
    manifest_df = pandas.read_csv(manifest_path, nrows=max_models)

    # ----- Load pseudosequences first so we can canonicalize -----
    allele_to_sequence = None
    allele_to_canonical = {}
    allele_sequences_filename = None
    candidates = pseudosequence_filename_candidates(models_dir)
    if candidates:
        allele_sequences_filename = candidates[0]

    if allele_sequences_filename is not None:
        allele_to_sequence = pandas.read_csv(
            join(models_dir, allele_sequences_filename),
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
            logging.debug(
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
        weights_filename = predictor_class.weights_path(
            models_dir, row.model_name)
        config = json.loads(row.config_json)

        model = Class1NeuralNetwork.from_config(
            config,
            weights_loader=partial(load_weights, abspath(weights_filename)),
            weight_paths=abspath(weights_filename))
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

    result = predictor_class(
        allele_to_allele_specific_models=allele_to_allele_specific_models,
        class1_pan_allele_models=class1_pan_allele_models,
        allele_to_sequence=allele_to_sequence,
        manifest_df=manifest_df,
        allele_to_percent_rank_transform=allele_to_percent_rank_transform,
        provenance_string=provenance_string,
        optimization_info=optimization_info,
        models_dir=abspath(models_dir),
    )
    if allele_to_sequence is not None:
        result.allele_to_canonical = allele_to_canonical
    if optimization_level >= 1:
        optimized = result.optimize()
        logging.info(
            "Model optimization %s",
            "succeeded" if optimized else "not supported for these models")
    return result
