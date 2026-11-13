# PYTHON_ARGCOMPLETE_OK
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

"""Entry points for the deprecated ``mhcflurry-*`` console scripts.

Each hyphenated training, selection and calibration script runs the same code
as ``mhcflurry <subcommand>``, after a one-line notice on stderr. They will be
removed in 2.4.0. ``mhcflurry-downloads``, ``mhcflurry-predict`` and
``mhcflurry-predict-scan`` are not deprecated and do not go through here.
"""

import importlib
import sys

from .completion import is_completing
from .main import _SUBCOMMANDS


def deprecated_script(subcommand):
    """
    Return the entry point of the console script ``mhcflurry-<subcommand>``.

    Parameters
    ----------
    subcommand : str
        Name of the ``mhcflurry`` subcommand the script duplicates.

    Returns
    -------
    callable
        Takes no arguments. Prints the deprecation notice to stderr, unless
        answering a shell completion request, then runs the subcommand on
        ``sys.argv[1:]``.
    """
    module_path, entry, _ = _SUBCOMMANDS[subcommand]

    def run():
        if not is_completing():
            print(
                "mhcflurry-%s is deprecated and will be removed in 2.4.0; "
                "use mhcflurry %s." % (subcommand, subcommand),
                file=sys.stderr)
        return getattr(importlib.import_module(module_path), entry)(sys.argv[1:])

    return run


calibrate_percentile_ranks = deprecated_script("calibrate-percentile-ranks")
class1_train_allele_specific_models = deprecated_script(
    "class1-train-allele-specific-models")
class1_select_allele_specific_models = deprecated_script(
    "class1-select-allele-specific-models")
class1_train_pan_allele_models = deprecated_script("class1-train-pan-allele-models")
class1_select_pan_allele_models = deprecated_script(
    "class1-select-pan-allele-models")
class1_train_processing_models = deprecated_script("class1-train-processing-models")
class1_select_processing_models = deprecated_script(
    "class1-select-processing-models")
class1_train_presentation_models = deprecated_script(
    "class1-train-presentation-models")
