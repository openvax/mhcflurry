#!/usr/bin/env python

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


from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals
)

from os.path import join
import argparse

import numpy as np

import keras
from keras.models import Graph
from keras.layers.core import Dense, Flatten
from keras.layers.noise import GaussianDropout
from keras.layers.embeddings import Embedding
from keras.utils import np_utils

from mhcflurry.data_helpers import load_allele_datasets
from mhcflurry.class1_allele_specific_hyperparameters import (
    EMBEDDING_DIM,
    HIDDEN_LAYER_SIZE,
    MAX_IC50
)
from mhcflurry.paths import (
    CLASS1_DATA_DIRECTORY
)
CSV_FILENAME = "combined_human_class1_dataset.csv"
CSV_PATH = join(CLASS1_DATA_DIRECTORY, CSV_FILENAME)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--binding-data-csv-path",
    default=CSV_PATH,
    help="CSV file with 'mhc', 'peptide', 'peptide_length', 'meas' columns")

parser.add_argument(
    "--min-samples-per-allele",
    default=5,
    help="Don't train predictors for alleles with fewer samples than this",
    type=int)

def build_graph_model(peptide_length=9,
                      amino_acid_vocabulary_size=20,
                      num_alleles=123,
                      init="he_uniform",
                      hidden_layer_size=HIDDEN_LAYER_SIZE,
                      output_activation="sigmoid",
                      activation="relu",
                      embedding_output_dim=EMBEDDING_DIM,
                      dropout_probability=0.1,
                      ):
    graph = Graph()
    graph.add_input(name='peptide', input_shape=(peptide_length,), dtype='int')

    graph.add_node(Embedding(input_dim=amino_acid_vocabulary_size, 
                             output_dim=embedding_output_dim, 
                             init=init,
                             input_length=peptide_length),
                   input='peptide', 
                   name='peptide_embedding')

    graph.add_node(Flatten(), 
                   input='peptide_embedding', 
                   name='flatten_peptide_embedding')

    graph.add_node(Dense(output_dim=hidden_layer_size,
                         activation=activation,
                         init=init), 
                   input='flatten_peptide_embedding', 
                   name='hidden_peptide')

    graph.add_node(GaussianDropout(dropout_probability), 
                   name='dropout_peptide',
                   input='hidden_peptide')

    graph.add_input(name='allele', input_shape=(num_alleles,))
    graph.add_node(Dense(
                         output_dim=hidden_layer_size,
                         activation=activation,
                         init=init), 
                   input='allele', 
                   name='hidden_allele')
    graph.add_node(GaussianDropout(dropout_probability), 
                   name='dropout_allele',
                   input='hidden_allele')

    graph.add_node(Dense(output_dim=hidden_layer_size, activation=activation,),
                   name='combined_dense', 
                   inputs=['dropout_peptide', 'dropout_allele'], 
                   merge_mode='concat')

    graph.add_node(Dense(output_dim=1, activation=output_activation),
                   name='output_dense', 
                   input='combined_dense')
    graph.add_output(name='output', input='output_dense')

    return graph

if __name__ == "__main__":
    args = parser.parse_args()

    allele_groups = load_allele_datasets(
        args.binding_data_csv_path,
        peptide_length=9,
        binary_encoding=False,
        max_ic50=MAX_IC50,
        sep=",",
        peptide_column_name="peptide")
    # concatenate datasets from all alleles to use for pre-training of
    # allele-specific predictors
    X_peptide = np.vstack([group.X for group in allele_groups.values()])
    Y_all = np.concatenate([group.Y for group in allele_groups.values()])
    print("Total Dataset size = %d" % len(Y_all))

    # Build 1-hot encoding of alleles
    allele_to_index = \
      dict((allele, i) 
        for (i, allele) in enumerate(allele_groups.keys()))

    X_allele_vector = np.array(
        [allele_to_index[allele] 
          for (allele, group) in allele_groups.items() 
          for x in range(group.Y.shape[0])])

    X_allele = np_utils.to_categorical(X_allele_vector)

    optimizer = keras.optimizers.RMSprop(
      lr=0.001,
      rho=0.9,
      epsilon=1e-6)

    graph = build_graph_model()

    print("Compiling model...")
    graph.compile(optimizer=optimizer, loss={'output': 'mse'})

    print("Fitting model...")
    graph.fit({
        'peptide': X_peptide, 
        'allele': X_allele, 
        'output': Y_all}, 
        nb_epoch=20,
        validation_split=0.1,
        shuffle=True)