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

import os


def add_release_provenance(rows, environ=os.environ):
    """Append release commit/workflow metadata when supplied by a launcher."""
    for label, variable in (
            ("git commit", "MHCFLURRY_RELEASE_GIT_COMMIT"),
            ("workflow id", "MHCFLURRY_RELEASE_WORKFLOW_ID")):
        value = environ.get(variable, "").strip()
        if value:
            rows.append((label, value))
    return rows
