"""The generic processing sweep alias never inherits the kernel-width design."""

import sys

import pytest

from mhcflurry.cli import train_command


SWEEP = "processing-hyperparameter-sweep"
ARGS = ["--out", "sweep", "--train-data", "train.csv.bz2"]


@pytest.fixture
def launches(monkeypatch):
    """Record training-script launches instead of running them."""
    calls = []

    def fake_call(argv, env):
        calls.append((argv, env["MHCFLURRY_CLI_PROG"]))
        return 0

    monkeypatch.setattr(train_command.subprocess, "call", fake_call)
    return calls


def test_hyperparameter_sweep_without_design_errors_before_launch(launches, capsys):
    with pytest.raises(SystemExit) as error:
        train_command.run_argv([SWEEP, *ARGS])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "mhcflurry train %s: error: --design is required" % SWEEP in stderr
    assert launches == []


@pytest.mark.parametrize("design", [
    ["--design", "training-recipe"],
    ["--design=ranking-confirmation"],
])
def test_hyperparameter_sweep_with_explicit_design_dispatches(launches, design):
    assert train_command.run_argv([SWEEP, *ARGS, *design]) == 0

    script = str(train_command._training_script_path(SWEEP))
    assert launches == [
        ([sys.executable, script, *ARGS, *design], "mhcflurry train " + SWEEP)]


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_hyperparameter_sweep_help_does_not_require_design(launches, help_flag):
    assert train_command.run_argv([SWEEP, help_flag]) == 0

    assert [argv[2:] for argv, _ in launches] == [[help_flag]]


def test_kernel_sweep_alias_keeps_kernel_width_default(launches):
    assert train_command.run_argv(["processing-kernel-sweep", *ARGS]) == 0

    script = str(train_command._training_script_path("processing-kernel-sweep"))
    assert launches == [
        ([sys.executable, script, *ARGS], "mhcflurry train processing-kernel-sweep")]
