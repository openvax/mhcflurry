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
    N_EPOCHS,
    INITIALIZATION_METHOD,
    ACTIVATION,
    BATCH_SIZE
)
from .class1_binding_predictor import Class1BindingPredictor
from .imputation_helpers import imputer_from_name
from .feedforward_hyperparameters import all_combinations_of_hyperparameters

from keras.optimizers import RMSprop

def add_imputation_argument_to_parser(parser, default="none"):
    """
    Extends an argument parser with --imputation-method
    """
    group = parser.add_argument_group("Imputation")
    group.add_argument(
        "--imputation-method",
        default=default,
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
    group = parser.add_argument_group("Neural Network Hyperparameters")
    group.add_argument(
        "--initialization",
        default=[INITIALIZATION_METHOD],
        nargs="+",
        help="Initialization for neural network weights Default: %(default)s")

    group.add_argument(
        "--activation",
        default=[ACTIVATION],
        choices=("tanh", "sigmoid", "relu", "linear", "softmax", "softplus"),
        nargs="+",
        help="Activation function for neural network layers."
        "Default: %(default)s")

    group.add_argument(
        "--embedding-size",
        type=int,
        default=[16, 32, 64],
        help="Size of vector representations for embedding amino acids. "
        "Default: %(default)s")

    group.add_argument(
        "--hidden-layer-size",
        type=int,
        default=[4, 16, 64],
        nargs="+",
        help="Size of hidden neural network layer. Default: %(default)s")

    group.add_argument(
        "--dropout",
        type=float,
        default=[0, 0.5],
        nargs="+",
        help="Dropout probability after neural network layers. "
        "Default: %(default)s")

    group.add_argument(
        "--kmer-size",
        type=int,
        default=9,
        help="Size of input vector for neural network")

    group.add_argument(
        "--max-ic50",
        type=float,
        default=[MAX_IC50],
        help="Largest IC50 represented by neural network output."
        "Default: %(default)s")

    group.add_argument(
        "--batch-normalization",
        default=[False],
        type=lambda s: bool(int(s)),
        nargs="+",
        help=(
            "Use batch normalization on layer activations, "
            "should be specified as integer values '0' or '1'."))
    return parser


def add_training_arguments_to_parser(parser):
    """
    Extends an argument parser with the following:
        --training-epochs
        --random-negative-samples
        --learning-rate
        --batch-size
    """
    group = parser.add_argument_group("Training Parameters")
    group.add_argument(
        "--random-negative-samples",
        type=int,
        nargs="+",
        default=[0, 10, 100],
        help="Number of random negtive samples to generate each training epoch")

    group.add_argument(
        "--learning-rate",
        type=float,
        default=[0.001],
        nargs="+",
        help="Learning rate for training neural network. Default: %(default)s")

    group.add_argument(
        "--training-epochs",
        type=int,
        default=N_EPOCHS,
        help="Number of training epochs. Default: %(default)s")

    group.add_argument(
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


def feedforward_parameter_grid_from_args(args):
    return all_combinations_of_hyperparameters(
        activation=args.activation,
        initialization_method=args.initialization,
        embedding_dim=args.embedding_size,
        dropout_probability=args.dropout,
        hidden_layer_size=args.hidden_layer_size,
        loss=["mse"],
        learning_rate=args.learning_rate,
        n_training_epochs=args.training_epochs,
        batch_size=args.batch_size,
        batch_normalization=args.batch_normalization)

def predictors_from_args(args, allele_name):
    """
    Given parsed arguments generates a sequence of Class1BindingPredictor
    objects
    """
    for params in feedforward_parameter_grid_from_args(args):
        yield Class1BindingPredictor.from_hyperparameters(
            name=allele_name,
            peptide_length=args.kmer_size,
            max_ic50=args.max_ic50,
            embedding_output_dim=args.embedding_size,
            layer_sizes=[params.hidden_layer_size],
            activation=args.activation,
            init=args.initialization,
            dropout_probability=args.dropout,
            optimizer=RMSprop(lr=args.learning_rate),
            batch_size=args.batch_size,
            batch_normalization=args.batch_normalization,
            n_random_negative_samples=args.random_negative_samples)

def predictor_from_args(args, allele_name):
    one_or_more_predictors = list(predictors_from_args(args, allele_name))
    if len(one_or_more_predictors) == 0:
        raise ValueError("No predictors created from given arguments!")
    if len(one_or_more_predictors) > 1:
        raise ValueError("Expected only one predictor, got %d" % (
            len(one_or_more_predictors)))
    return one_or_more_predictors[0]

def imputer_from_args(args):
    return imputer_from_name(args.imputation_method)
