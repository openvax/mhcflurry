#!/usr/bin/env python

"""
Train one neural network for every allele w/ more than 50 data points in
our dataset.

Using the following hyperparameters:
    embedding_size=64,
    layer_sizes=(400,),
    activation='relu',
    loss='mse',
    init='lecun_uniform',
    n_pretrain_epochs=10,
    n_epochs=100,
    dropout_probability=0.25
...which performed well in held out average AUC across alleles in the
Nielsen 2009 dataset.
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals
)
from shutil import rmtree
from os import makedirs
from os.path import exists, join
import argparse

import numpy as np

from mhcflurry.common import normalize_allele_name
from mhcflurry.feedforward import make_network
from mhcflurry.data_helpers import load_data
from mhcflurry.class1_allele_specific_hyperparameters import (
    N_PRETRAIN_EPOCHS,
    N_EPOCHS,
    ACTIVATION,
    INITIALIZATION_METHOD,
    EMBEDDING_DIM,
    HIDDEN_LAYER_SIZE,
    DROPOUT_PROBABILITY
)
from mhcflurry.paths import (
    CLASS1_MODEL_DIRECTORY,
    CLASS1_DATA_DIRECTORY
)
CSV_FILENAME = "combined_human_class1_dataset.csv"
CSV_PATH = join(CLASS1_DATA_DIRECTORY, CSV_FILENAME)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output-dir",
    default=CLASS1_MODEL_DIRECTORY,
    help="Output directory for allele-specific predictor HDF weights files")

parser.add_argument(
    "--overwrite",
    default=False,
    action="store_true",
    help="Overwrite existing output directory")

parser.add_argument(
    "--binding-data-csv-path",
    default=CSV_PATH,
    help="CSV file with 'mhc', 'peptide', 'peptide_length', 'meas' columns")

if __name__ == "__main__":
    args = parser.parse_args()

    if exists(args.output_dir):
        if args.overwrite:
            rmtree(args.output_dir)
    else:
        makedirs(args.output_dir)
    allele_groups, _ = load_data(
        args.binding_data_csv_path,
        peptide_length=9,
        binary_encoding=False,
        sep=",",
        peptide_column_name="peptide")
    # concatenate datasets from all alleles to use for pre-training of
    # allele-specific predictors
    X_all = np.vstack([group.X for group in allele_groups.values()])
    Y_all = np.concatenate([group.Y for group in allele_groups.values()])
    print("Total Dataset size = %d" % len(Y_all))

    model = make_network(
        input_size=9,
        embedding_input_dim=20,
        embedding_output_dim=EMBEDDING_DIM,
        layer_sizes=(HIDDEN_LAYER_SIZE,),
        activation=ACTIVATION,
        init=INITIALIZATION_METHOD,
        dropout_probability=DROPOUT_PROBABILITY)
    print("Model config: %s" % (model.get_config(),))
    model.fit(X_all, Y_all, nb_epoch=N_PRETRAIN_EPOCHS)
    old_weights = model.get_weights()
    for allele_name, allele_data in allele_groups.items():
        allele_name = normalize_allele_name(allele_name)
        if allele_name.isdigit():
            print("Skipping allele %s" % (allele_name,))
            continue
        n_allele = len(allele_data.Y)
        print("%s: total count = %d" % (allele_name, n_allele))
        filename = allele_name + ".hdf"
        path = join(args.output_dir, filename)
        if exists(path) and not args.overwrite:
            print("-- already exists, skipping")
            continue
        if n_allele < 10:
            print("-- too few data points, skipping")
            continue
        model.set_weights(old_weights)
        model.fit(
            allele_data.X,
            allele_data.Y,
            nb_epoch=N_EPOCHS,
            show_accuracy=True)
        print("Saving model for %s to %s" % (allele_name, path))
        model.save_weights(path)
