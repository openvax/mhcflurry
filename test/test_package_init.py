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

"""Tests for package-import initialization."""

import os
import pathlib
import subprocess
import sys


PACKAGE_INIT = pathlib.Path(__file__).parents[1] / "mhcflurry" / "__init__.py"


def _mkl_threading_layer_after_startup(platform_name, initial=None):
    env = dict(os.environ)
    if initial is None:
        env.pop("MKL_THREADING_LAYER", None)
    else:
        env["MKL_THREADING_LAYER"] = initial
    code = """
import os
import runpy
import sys

sys.platform = %r
try:
    runpy.run_path(%r)
except ImportError:
    pass
print(os.environ.get("MKL_THREADING_LAYER", "<unset>"))
""" % (platform_name, str(PACKAGE_INIT))
    return subprocess.check_output(
        [sys.executable, "-c", code], env=env, text=True).strip()


def test_default_mkl_threading_layer_is_linux_only():
    assert _mkl_threading_layer_after_startup("win32") == "<unset>"
    assert _mkl_threading_layer_after_startup("darwin") == "<unset>"
    assert _mkl_threading_layer_after_startup("linux") == "GNU"
    assert _mkl_threading_layer_after_startup("linux", "TBB") == "TBB"
