# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Release provenance fields embedded in trained model metadata."""

import csv
from getpass import getuser
import os
from socket import gethostname
import time

from .version import __version__


def add_release_provenance(rows, environ=os.environ):
    """Append release commit/workflow metadata when supplied by a launcher."""
    for label, variable in (
            ("git commit", "MHCFLURRY_RELEASE_GIT_COMMIT"),
            ("workflow id", "MHCFLURRY_RELEASE_WORKFLOW_ID")):
        value = environ.get(variable, "").strip()
        if value:
            rows.append((label, value))
    return rows


def write_info_txt(models_dir, environ=None):
    """Write ``info.txt`` provenance metadata into ``models_dir``.

    Parameters
    ----------
    models_dir : str or os.PathLike
        Directory where ``info.txt`` will be written.
    environ : Mapping[str, str], optional
        Environment mapping for optional release provenance fields. Defaults
        to ``os.environ``.

    Returns
    -------
    str
        Path to the written ``info.txt`` file.
    """
    info_path = os.path.join(models_dir, "info.txt")
    rows = [
        ("trained on", time.asctime()),
        ("package   ", "mhcflurry %s" % __version__),
        ("hostname  ", gethostname()),
        ("user      ", getuser()),
    ]
    add_release_provenance(
        rows,
        environ=os.environ if environ is None else environ,
    )
    with open(info_path, "w", newline="") as fd:
        writer = csv.writer(fd, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)
    return info_path
