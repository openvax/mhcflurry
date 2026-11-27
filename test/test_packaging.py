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
import tomllib
from pathlib import Path

import setuptools
from packaging.requirements import Requirement
from packaging.version import Version

from mhcflurry.version import __version__

REPO_DIR = Path(__file__).resolve().parents[1]


def load_pyproject():
    """Load the repository's ``pyproject.toml``."""
    return tomllib.loads((REPO_DIR / "pyproject.toml").read_text(encoding="utf-8"))


def declared_console_scripts():
    """Return the console scripts declared in ``pyproject.toml``."""
    scripts = load_pyproject().get("project", {}).get("scripts", {})
    assert scripts, "pyproject.toml declares no [project.scripts]"
    return dict(scripts)


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


def test_pyproject_packages_cli_subpackage():
    pyproject = load_pyproject()
    find_config = pyproject["tool"]["setuptools"]["packages"]["find"]
    packages = setuptools.find_packages(
        where=str(REPO_DIR),
        include=find_config["include"],
        exclude=find_config.get("exclude", []),
    )
    assert "mhcflurry" in packages
    assert "mhcflurry.cli" in packages
    # Guard against ``find_packages()`` picking up the repo's ``test`` dir
    # (it has an ``__init__.py``) and shipping it as a top-level package.
    assert not any(
        p == "test" or p.startswith("test.") for p in packages
    ), "pyproject.toml must not ship the test package: %r" % packages
    assert "matplotlib" in pyproject["project"]["dependencies"]
    assert "downloads.yml" in pyproject["tool"]["setuptools"]["package-data"]["mhcflurry"]
    assert (
        pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "mhcflurry.version.__version__"
    )


def test_requirements_txt_matches_pyproject_dependencies():
    pyproject_deps = {
        str(Requirement(spec))
        for spec in load_pyproject()["project"]["dependencies"]
    }
    req_lines = [
        line.strip()
        for line in (REPO_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    req_deps = {str(Requirement(spec)) for spec in req_lines}
    assert req_deps == pyproject_deps


def test_no_biopython_in_runtime_or_downloads_generation():
    for root in (REPO_DIR / "mhcflurry", REPO_DIR / "downloads-generation"):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in (".py", ".txt"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "Bio.SeqIO" not in text, f"Biopython import found in {path}"
            assert "from Bio " not in text, f"Biopython import found in {path}"
            if path.name == "requirements.txt":
                assert "biopython" not in text.lower(), f"biopython listed in {path}"
