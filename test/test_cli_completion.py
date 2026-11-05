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

"""Shell tab completion (argcomplete) stays wired up and import-light."""

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mhcflurry.cli.completion import completing_subcommand, is_completing
from mhcflurry.cli.main import FLAG_COMPLETION_SUBCOMMANDS, _SUBCOMMANDS

REPO_DIR = Path(__file__).resolve().parents[1]

# Modules a Tab press must not import. ``yaml`` stands for reading the
# download catalogue, which costs tens of milliseconds.
HEAVY_MODULES = (
    "torch", "numpy", "pandas", "sklearn", "scipy", "matplotlib",
    "mhcgnomes", "tqdm", "yaml",
)


def console_scripts():
    """Read the console scripts that setup.py installs.

    Returns
    -------
    dict
        Script name mapped to its ``"module:function"`` entry point.
    """
    tree = ast.parse((REPO_DIR / "setup.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "console_scripts":
                    return dict(
                        (part.strip() for part in spec.split("=", 1))
                        for spec in ast.literal_eval(value))
    raise AssertionError("setup.py declares no console_scripts")


# The underscore-prefixed cluster worker is internal; nobody types it.
USER_SCRIPTS = {
    name: target for name, target in console_scripts().items()
    if not name.startswith("_")
}


def completion_environ(comp_line, **extra):
    """Build argcomplete's environment for a Tab press at the end of a line.

    Parameters
    ----------
    comp_line : str
        Command line typed so far.
    **extra
        Further variables, overriding the defaults.

    Returns
    -------
    dict
    """
    environ = {
        "_ARGCOMPLETE": "1",
        "COMP_LINE": comp_line,
        "COMP_POINT": str(len(comp_line)),
        "_ARGCOMPLETE_IFS": "\n",
    }
    environ.update(extra)
    return environ


def complete(target, comp_line, tmp_path):
    """Run a console-script entry point as one Tab press would.

    Parameters
    ----------
    target : str
        Entry point, ``"module:function"``.
    comp_line : str
        Command line typed so far; the cursor is at its end.
    tmp_path : pathlib.Path
        Directory for the completions file.

    Returns
    -------
    set of str
        The completions offered.
    """
    module, function = target.split(":")
    out = tmp_path / "completions.txt"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MHCFLURRY_")}
    env["PYTHONPATH"] = str(REPO_DIR)
    env.update(completion_environ(
        comp_line, _ARGCOMPLETE_STDOUT_FILENAME=str(out)))
    script = "import sys; from %s import %s; sys.exit(%s())" % (
        module, function, function)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=str(REPO_DIR),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return set(out.read_text().split("\n")) - {""}


def test_console_scripts_read_from_setup():
    assert USER_SCRIPTS["mhcflurry"] == "mhcflurry.cli.main:main"
    assert USER_SCRIPTS["mhcflurry-predict"] == "mhcflurry.cli.predict_command:run"


@pytest.mark.parametrize("name", sorted(USER_SCRIPTS))
def test_entry_module_carries_argcomplete_marker(name):
    # Global completion (activate-global-python-argcomplete) only completes
    # scripts whose entry module has the marker in its first 1024 bytes.
    module = USER_SCRIPTS[name].split(":")[0]
    path = REPO_DIR / (module.replace(".", "/") + ".py")
    assert "PYTHON_ARGCOMPLETE_OK" in path.read_bytes()[:1024].decode()


def test_is_completing():
    assert is_completing({"_ARGCOMPLETE": "1"})
    assert not is_completing({})


@pytest.mark.parametrize("comp_line,extra,expected", [
    ("mhcflurry predict --", {}, "predict"),
    ("mhcflurry predict ", {}, "predict"),
    ("mhcflurry downloads fetch --", {}, "downloads"),
    # The cursor is still on the subcommand word: complete its name.
    ("mhcflurry predict", {}, None),
    ("mhcflurry ", {}, None),
    ("mhcflurry not-a-command --", {}, None),
    ("python -m mhcflurry.cli.main predict --", {"_ARGCOMPLETE": "3"}, "predict"),
])
def test_completing_subcommand(comp_line, extra, expected):
    environ = completion_environ(comp_line, **extra)
    assert completing_subcommand(_SUBCOMMANDS, environ) == expected


def test_completing_subcommand_without_completion_request():
    assert completing_subcommand(_SUBCOMMANDS, {"COMP_LINE": "mhcflurry predict --"}) is None


@pytest.mark.parametrize(
    "subcommand",
    list(FLAG_COMPLETION_SUBCOMMANDS) + [
        "class1-train-pan-allele-models", "calibrate-percentile-ranks",
        "eval", "train", None])
def test_tab_press_imports_no_heavy_modules(subcommand):
    comp_line = "mhcflurry %s --" % subcommand if subcommand else "mhcflurry "
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MHCFLURRY_")}
    env["PYTHONPATH"] = str(REPO_DIR)
    env.update(completion_environ(comp_line))
    flag_completion = subcommand in FLAG_COMPLETION_SUBCOMMANDS
    module = _SUBCOMMANDS[subcommand][0] if subcommand else None
    script = '''
import sys
from mhcflurry.cli.main import build_completion_parser
build_completion_parser()
loaded = set(sys.modules)
heavy = loaded.intersection(%r)
assert not heavy, heavy
if %r:
    assert (%r in loaded) == %r, sorted(m for m in loaded if m.startswith("mhcflurry"))
''' % (HEAVY_MODULES, module, module, flag_completion)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=str(REPO_DIR),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("comp_line,expected", [
    ("mhcflurry ", {"predict", "predict-scan", "downloads", "train", "eval",
                    "class1-train-pan-allele-models", "pseudosequences"}),
    ("mhcflurry pred", {"predict", "predict-scan"}),
    ("mhcflurry predict --al", {"--alleles", "--allele-column"}),
    ("mhcflurry predict --model", {"--models", "--model-release"}),
    ("mhcflurry predict-scan --seq", {"--sequences", "--sequence-id-column",
                                      "--sequence-column"}),
    ("mhcflurry downloads ", {"fetch", "info", "list", "releases", "path", "url"}),
    ("mhcflurry downloads fetch --", {"--release", "--keep",
                                      "--already-downloaded-dir"}),
    ("mhcflurry pseudosequences ", {"filename", "legacy", "path", "list"}),
])
def test_mhcflurry_completes_subcommands_and_flags(comp_line, expected, tmp_path):
    completions = complete("mhcflurry.cli.main:main", comp_line, tmp_path)
    assert expected <= completions, completions


def test_mhcflurry_completes_other_subcommands_by_name_only(tmp_path):
    # Their modules import torch or pandas at module level, which would cost
    # seconds per Tab press.
    completions = complete(
        "mhcflurry.cli.main:main",
        "mhcflurry class1-train-pan-allele-models --", tmp_path)
    assert completions == set()


@pytest.mark.parametrize("name", sorted(set(USER_SCRIPTS) - {"mhcflurry"}))
def test_historical_scripts_complete_their_flags(name, tmp_path):
    completions = complete(USER_SCRIPTS[name], "%s -" % name, tmp_path)
    assert "--help" in completions, completions


def test_downloads_import_defers_reading_the_catalogue():
    import yaml

    catalogue = yaml.safe_load((REPO_DIR / "mhcflurry/downloads.yml").read_text())
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MHCFLURRY_")}
    env["PYTHONPATH"] = str(REPO_DIR)
    script = '''
import sys
from mhcflurry import downloads
assert "yaml" not in sys.modules
assert downloads.get_current_release() == %r
assert "yaml" in sys.modules
assert downloads.get_downloads_dir().endswith(%r)
''' % (catalogue["current-release"], catalogue["current-release"])
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=str(REPO_DIR),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
