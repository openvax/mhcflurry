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
Helper functions for serialization/deserialization of Keras models
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import logging
from os.path import exists
from os import remove
import json

from keras.models import model_from_config

from .feedforward import compile_network

def load_keras_model_from_disk(
        model_json_path,
        weights_hdf_path,
        name=None):
    """
    Loads a model from two files on disk: a JSON configuration and HDF5 weights.

    Parameters
    ----------
    model_json_path : str

    weights_hdf_path : str

    name : str, optional

    Returns a Keras model.
    """

    if not exists(model_json_path):
        raise ValueError("Model file %s (name = %s) not found" % (
            model_json_path, name,))

    with open(model_json_path, "r") as f:
        config_dict = json.load(f)

    model = model_from_config(config_dict)

    if weights_hdf_path is not None:
        if not exists(weights_hdf_path):
            raise ValueError(
                "Missing model weights file %s (name = %s)" % (weights_hdf_path, name))
        model.load_weights(weights_hdf_path)

    # In some cases I haven't been able to use a model after loading it
    # without compiling it first.
    compile_network(model)
    return model


def save_keras_model_to_disk(
        model,
        model_json_path,
        weights_hdf_path,
        overwrite=False):
    """
    Write a Keras model configuration to disk along with its weights.

    Parameters
    ----------
    model : keras model

    model_json_path : str
        Path to JSON file where model configuration will be saved.

    weights_hdf_path : str
        Path to HDF5 file where model weights will be saved.

    overwrite : bool
        Overwrite existing files?
    """
    if exists(model_json_path) and overwrite:
        logging.info(
            "Removing existing model JSON file '%s'" % (
                model_json_path,))
        remove(model_json_path)

    if exists(model_json_path):
        logging.warn(
            "Model JSON file '%s' already exists" % (model_json_path,))
    else:
        logging.info(
            "Saving model file %s" % (model_json_path,))
        with open(model_json_path, "w") as f:
            config = model.get_config()
            json.dump(config, f)

    if exists(weights_hdf_path) and overwrite:
        logging.info(
            "Removing existing model weights HDF5 file '%s'" % (
                weights_hdf_path,))
        remove(weights_hdf_path)

    if exists(weights_hdf_path):
        logging.warn(
            "Model weights HDF5 file '%s' already exists" % (weights_hdf_path,))
    else:
        logging.info(
            "Saving model weights HDF5 file %s" % (weights_hdf_path,))
        model.save_weights(weights_hdf_path)
