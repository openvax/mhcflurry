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

"""Packaging metadata tests."""

import runpy
from pathlib import Path

import setuptools


def test_setup_packages_cli_subpackage(monkeypatch):
    captured = {}
    repo_dir = Path(__file__).resolve().parents[1]

    def fake_setup(**kwargs):
        captured.update(kwargs)

    monkeypatch.chdir(repo_dir)
    monkeypatch.setattr(setuptools, "setup", fake_setup)
    runpy.run_path(str(repo_dir / "setup.py"), run_name="__main__")

    packages = captured["packages"]
    assert "mhcflurry" in packages
    assert "mhcflurry.cli" in packages
    # Guard against ``find_packages()`` picking up the repo's ``test`` dir
    # (it has an ``__init__.py``) and shipping it as a top-level package.
    assert not any(
        p == "test" or p.startswith("test.") for p in packages
    ), "setup.py must not ship the test package: %r" % packages
