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

import json
import re
import runpy
from pathlib import Path

import setuptools
from packaging.requirements import Requirement
from packaging.version import Version

from mhcflurry.version import __version__


def test_install_guidance_targets_current_stable_series():
    repo_dir = Path(__file__).resolve().parents[1]
    prerelease = re.compile(r"\b\d+\.\d+\.\d+(?:a|b|rc)\d+\b")
    version = Version(__version__)
    series = '%d.%d' % (version.major, version.minor)

    def check_installation(text):
        requirements = re.findall(r'%?pip install --upgrade "(mhcflurry[^\"]+)"', text)
        assert requirements
        for requirement in requirements:
            specifier = Requirement(requirement).specifier
            assert specifier.contains(version, prereleases=False)
            assert specifier.contains('%s.%d' % (series, version.micro + 1), prereleases=False)
            assert not specifier.contains('%d.%d.0' % (version.major, version.minor + 1))
            if version.minor:
                assert not specifier.contains('%d.%d.0' % (version.major, version.minor - 1))
        # Check the introduction too: a correct command can still sit beneath
        # prose advertising a different patch, as happened in README/Colab.
        for line in text.splitlines():
            if 'install' in line.lower():
                advertised = re.search(r'\bMHCflurry (\d+(?:\.\d+)*)', line)
                if advertised:
                    assert advertised.group(1) == series
        assert '--pre ' not in text
        assert not prerelease.search(text)

    for relative_path in ("README.md", "docs/intro.md"):
        text = (repo_dir / relative_path).read_text()
        check_installation(text)

    notebook = json.loads(
        (repo_dir / "notebooks/mhcflurry-colab.ipynb").read_text())
    setup_cell = next(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "".join(setup_cell["source"])
    check_installation('\n'.join(''.join(cell['source']) for cell in notebook['cells']))
    assert "mhcflurry-downloads --quiet fetch models_class1_presentation" in source


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
    assert "matplotlib" in captured["install_requires"]
