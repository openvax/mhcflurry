#!/usr/bin/env python
#
# Copyright (c) 2015. Mount Sinai School of Medicine
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

import argparse

import numpy as np

import mhcflurry
from mhcflurry.data import (
    load_allele_datasets,
    encode_peptide_to_affinity_dict,
)


from arg_parsing import parse_int_list, parse_float_list
from dataset_paths import PETERS2009_CSV_PATH
from common import load_csv_binding_data_as_dict
from training_helpers import kfold_cross_validation_of_model_params_with_synthetic_data

parser = argparse.ArgumentParser()

parser.add_argument(
    "--binding-data-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--synthetic-data-csv",
    required=True,
    help="CSV with {mhc, sequence, ic50} columns of synthetic data")

parser.add_argument(
    "--max-ic50",
    default=50000.0,
    type=float)

parser.add_argument(
    "--embedding-dim-sizes",
    default=[4, 8, 16, 32, 64],
    type=parse_int_list)

parser.add_argument(
    "--hidden-layer-sizes",
    default=[4, 8, 16, 32, 64, 128, 256],
    type=parse_int_list)

parser.add_argument(
    "--dropouts",
    default=[0.0, 0.25, 0.5],
    type=parse_float_list)

parser.add_argument(
    "--activation-functions",
    default=["tanh", "relu"],
    type=lambda s: [si.strip() for si in s.split(",")])

parser.add_argument(
    "--training-epochs",
    type=int,
    default=250)

parser.add_argument("--log-file")


def get_extra_data(allele, train_datasets, expanded_predictions):
    original_dataset = train_datasets[allele]
    original_peptides = set(original_dataset.peptides)
    expanded_allele_affinities = expanded_predictions[allele]
    extra_affinities = {
        k: v
        for (k, v) in expanded_allele_affinities.items()
        if k not in original_peptides
    }
    extra_peptides = list(extra_affinities.keys())
    extra_values = list(extra_affinities.values())
    extra_X = mhcflurry.data_helpers.index_encoding(extra_peptides, peptide_length=9)
    extra_Y = np.array(extra_values)
    return extra_X, extra_Y


def rescale_ic50(ic50, max_ic50):
    log_ic50 = np.log(ic50) / np.log(args.max_ic50)
    return max(0.0, min(1.0, 1.0 - log_ic50))


def load_synthetic_data(csv_path, max_ic50):
    synthetic_allele_to_peptide_to_ic50_dict = load_csv_binding_data_as_dict(
        csv_path)
    return {
        allele: {
            peptide: rescale_ic50(ic50, max_ic50=max_ic50)
            for (peptide, ic50)
            in allele_dict.items()
        }
        for (allele, allele_dict)
        in synthetic_allele_to_peptide_to_ic50_dict.items()
    }


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    print("Loading binding data from %s" % args.binding_data_csv)
    allele_datasets = load_allele_datasets(
        args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=False)
    print("Loading synthetic data from %s" % args.synthetic_data_csv)

    synthetic_affinities = load_synthetic_data(
        csv_path=args.synthetic_data_csv,
        max_ic50=args.max_ic50)

    combined_allele_set = set(allele_datasets.keys()).union(
        synthetic_affinities.keys())

    combined_allele_list = list(sorted(combined_allele_set))

    if args.log_file:
        logfile = open(args.log_file, "w")
        logfile.write("allele,n_samples,n_unique,n_synth,")
    else:
        logfile = None
    logfile_needs_header = True

    for allele in combined_allele_list:
        synthetic_allele_dict = synthetic_affinities[allele]
        (_, _, Counts_synth, X_synth, _, Y_synth) = \
            encode_peptide_to_affinity_dict(synthetic_allele_dict)
        synthetic_sample_weights = 1.0 / Counts_synth
        scores = {}
        source_peptides = allele_datasets[allele].original_peptides
        X_original = allele_datasets[allele].X_index
        Y_original = allele_datasets[allele].Y
        n_samples = len(X_original)
        n_unique_samples = len(set(source_peptides))
        n_synth_samples = len(synthetic_sample_weights)
        for dropout in args.dropouts:
            for embedding_dim_size in args.embedding_dim_sizes:
                for hidden_layer_size in args.hidden_layer_sizes:
                    for activation in args.activation_functions:
                        params = (
                            ("dropout_probability", dropout),
                            ("embedding_dim_size", embedding_dim_size),
                            ("hidden_layer_size", hidden_layer_size),
                            ("activation", activation),
                        )

                        print(
                            "Evaluating allele %s (n=%d, unique=%d): %s" % (
                                allele,
                                n_samples,
                                n_unique_samples,
                                params))
                        average_scores, _ = \
                            kfold_cross_validation_of_model_params_with_synthetic_data(
                                X_original=X_original,
                                Y_original=Y_original,
                                source_peptides_original=source_peptides,
                                X_synth=X_synth,
                                Y_synth=Y_synth,
                                original_sample_weights=allele_datasets[allele].weights,
                                synthetic_sample_weights=synthetic_sample_weights,
                                n_training_epochs=args.training_epochs,
                                max_ic50=args.max_ic50,
                                **dict(params))
                        if logfile:
                            if logfile_needs_header:
                                for param_name, _ in params:
                                    logfile.write("%s," % param_name)
                                for score_name in average_scores._fields:
                                    logfile.write("%s," % score_name)
                                logfile.write("param_id\n")
                                logfile_needs_header = False
                            logfile.write("%s,%d,%d,%d," % (
                                allele,
                                n_samples,
                                n_unique_samples,
                                n_synth_samples))
                            for _, param_value in params:
                                logfile.write("%s," % param_value)
                            for score_name in average_scores._fields:
                                score_value = average_scores.__dict__[score_name]
                                logfile.write("%0.4f," % score_value)
                            logfile.write("%d\n" % len(scores))
                            logfile.flush()

                        scores[params] = average_scores
                        print("%s => %s" % (params, average_scores))
    if logfile:
        logfile.close()
