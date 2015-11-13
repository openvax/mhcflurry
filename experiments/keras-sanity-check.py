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

"""
Make sure a Keras model returns the same output before and after serialization
"""

import json

from keras.models import Sequential, model_from_config
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding

import numpy as np
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--only-load", default=False, action="store_true")


def encode(X, embedding_matrix):
    n_samples, input_length = X.shape
    embedding_dim = embedding_matrix.shape[1]

    X_embedded = np.zeros((n_samples, input_length * embedding_dim))
    for i in range(n_samples):
        for j in range(input_length):
            symbol_idx = X_train_index[i, j]
            X_embedded[i, j * embedding_dim:(j + 1) * embedding_dim] = \
                embedding_matrix[symbol_idx, :]
    return X_embedded


if __name__ == "__main__":

    args = parser.parse_args()

    n_symbols = 3
    input_length = 2
    n_training_samples = 5000
    n_test_samples = 2
    embedding_dim = 2

    X_train_index = np.random.randint(
        low=0,
        high=n_symbols,
        size=(n_training_samples, input_length))
    # embed into 15 dimensions

    embedding_matrix = np.random.randn(n_symbols, embedding_dim)

    X_train_embedded = encode(X_train_index, embedding_matrix)
    w = np.random.randn(embedding_dim * input_length)
    Y_train = X_train_embedded.dot(w)
    X_test_index = np.random.randint(
        low=0,
        high=n_symbols,
        size=(n_test_samples, input_length))
    print("X_test", X_test_index)
    X_test_embedded = encode(X_test_index, embedding_matrix)

    json_path = "keras-sanity-check.json"
    hdf_path = "keras-sanity-check.hdf"

    if not args.only_load:
        Y_test = X_test_embedded.dot(w)
        print("Y_test", Y_test)

        model = Sequential()
        model.add(Embedding(
            input_length=input_length,
            input_dim=n_symbols,
            output_dim=embedding_dim))
        model.add(Flatten())
        model.add(Dropout(p=0.25))
        model.add(Dense(
            input_dim=embedding_dim * input_length,
            output_dim=1, activation="linear"))
        model.compile(loss="mse", optimizer="sgd")
        model.fit(X_train_index, Y_train, verbose=0)
        print("model weights before", model.get_weights())
        pred_before = model.predict(X_test_index)
        print("pred_before", pred_before)

        with open(json_path, "w") as f:
            f.write(model.to_json())

        model.save_weights(hdf_path, overwrite=True)

    with open(json_path, "r") as f:
        json_dict = json.load(f)

    model2 = model_from_config(json_dict)
    print(
        "weights before load",
        model2.get_weights())
    model2.load_weights(hdf_path)
    print(
        "weights after load",
        model2.get_weights())

    print("pred after load", model2.predict(X_test_index))
