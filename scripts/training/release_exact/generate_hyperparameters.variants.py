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
Generate grid of hyperparameters.
"""
import argparse

from mhcflurry.cli.generate_training_hyperparameters import (
    PROCESSING_VARIANT_CHOICES,
    build_processing_variant_grid,
    run_processing_variant_argv as main,
    transform_processing_hyperparameters,
)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "production_hyperparameters",
        metavar="YAML",
        help="Production processing hyperparameter grid.")
    parser.add_argument(
        "kind",
        choices=PROCESSING_VARIANT_CHOICES,
        help="Hyperparameters variant to output.")
    return parser


def transform(kind, hyperparameters):
    return [transform_processing_hyperparameters(kind, hyperparameters)]


def build_grid(production_hyperparameters, kind):
    return build_processing_variant_grid(production_hyperparameters, kind)


if __name__ == "__main__":
    main()
