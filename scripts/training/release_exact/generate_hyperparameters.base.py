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
from mhcflurry.cli.generate_training_hyperparameters import (
    PROCESSING_DEFAULT_MINIBATCH_SIZE,
    add_minibatch_argument,
    build_processing_base_grid as build_grid,
    processing_base_grid_iter as hyperparameters_grid,
    run_processing_base_argv as main,
)


def make_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_minibatch_argument(parser, PROCESSING_DEFAULT_MINIBATCH_SIZE)
    return parser


if __name__ == "__main__":
    main()
