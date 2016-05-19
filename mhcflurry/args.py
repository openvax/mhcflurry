# Copyright (c) 2016. Mount Sinai School of Medicine
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

from .regression_target import MAX_IC50
from .feedforward_hyperparameters import (
    HIDDEN_LAYER_SIZE,
    EMBEDDING_DIM,
    N_EPOCHS,
    INITIALIZATION_METHOD,
    ACTIVATION,
    DROPOUT_PROBABILITY,
    BATCH_SIZE
)
from .class1_binding_predictor import Class1BindingPredictor
from .imputation_helpers import imputer_from_name

from keras.optimizers import RMSprop

def add_imputation_argument_to_parser(parser):
    """
    Extends an argument parser with --imputation-method
    """
    parser.add_argument(
        "--imputation-method",
        default="none",
        choices=("mice", "knn", "softimpute", "svd", "mean", "none"),
        type=lambda s: s.strip().lower(),
        help="Use the given imputation method to generate data for pre-training models")
    return parser


def add_hyperparameter_arguments_to_parser(parser):
    """
    Extend an argument parser with the following options:
        --activation
        --initialization
        --embedding-size
        --hidden-layer-size
        --dropout
        --kmer-size

    """
    parser.add_argument(
        "--initialization",
        default=INITIALIZATION_METHOD,
        help="Initialization for neural network weights Default: %(default)s")

    parser.add_argument(
        "--activation",
        default=ACTIVATION,
        help="Activation function for neural network layers. "
        "Default: %(default)s")

    parser.add_argument(
        "--embedding-size",
        type=int,
        default=EMBEDDING_DIM,
        help="Size of vector representations for embedding amino acids. "
        "Default: %(default)s")

    parser.add_argument(
        "--hidden-layer-size",
        type=int,
        default=HIDDEN_LAYER_SIZE,
        help="Size of hidden neural network layer. Default: %(default)s")

    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT_PROBABILITY,
        help="Dropout probability after neural network layers. "
        "Default: %(default)s")

    parser.add_argument(
        "--kmer-size",
        type=int,
        default=9,
        help="Size of input vector for neural network")

    parser.add_argument(
        "--max-ic50",
        type=float,
        default=MAX_IC50,
        help="Largest IC50 represented by neural network output. "
        "Default: %(default)s")
    return parser

def add_training_arguments_to_parser(parser):
    """
    Extends an argument parser with the following:
        --training-epochs
        --random-negative-samples
        --learning-rate
        --batch-size
    """
    parser.add_argument(
        "--random-negative-samples",
        type=int,
        default=0,
        help="Number of random negtive samples to generate each training epoch")

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for training neural network. Default: %(default)s")

    parser.add_argument(
        "--training-epochs",
        type=int,
        default=N_EPOCHS,
        help="Number of training epochs. Default: %(default)s")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of samples in SGD mini-batch")

    return parser

def add_arguments_to_parser(parser):
    """
    Extend an argument parser with the following options:
        --training-epochs
        --activation
        --initialization
        --embedding-size
        --hidden-layer-size
        --dropout
        --max-ic50
        --random-negative-samples
        --imputation-method
        --learning-rate
        --kmer-size
    """
    functions = [
        add_hyperparameter_arguments_to_parser,
        add_training_arguments_to_parser,
        add_imputation_argument_to_parser,
    ]
    for fn in functions:
        parser = fn(parser)
    return parser


def predictor_from_args(args, allele_name):
    """
    Given parsed arguments returns a Class1BindingPredictor
    """
    layer_sizes = (args.hidden_layer_size,) if args.hidden_layer_size > 0 else ()
    return Class1BindingPredictor.from_hyperparameters(
        name=allele_name,
        peptide_length=args.kmer_size,
        max_ic50=args.max_ic50,
        embedding_output_dim=args.embedding_size,
        layer_sizes=layer_sizes,
        activation=args.activation,
        init=args.initialization,
        dropout_probability=args.dropout,
        optimizer=RMSprop(lr=args.learning_rate))

def imputer_from_args(args):
    return imputer_from_name(args.imputation_method)
