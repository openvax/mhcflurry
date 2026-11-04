"""Tests for the shared argparse helpers in :mod:`mhcflurry.common`."""

import argparse

import pytest

from mhcflurry.common import (
    add_debug_arg,
    add_max_epochs_arg,
    add_out_models_dir_arg,
    add_verbosity_arg,
)


def test_add_verbosity_arg_defaults_and_override():
    parser = argparse.ArgumentParser()
    add_verbosity_arg(parser)
    assert parser.parse_args([]).verbosity == 0
    assert parser.parse_args(["--verbosity", "2"]).verbosity == 2

    parser = argparse.ArgumentParser()
    add_verbosity_arg(parser, default=1)
    assert parser.parse_args([]).verbosity == 1


def test_add_debug_arg():
    parser = argparse.ArgumentParser()
    add_debug_arg(parser)
    assert parser.parse_args([]).debug is False
    assert parser.parse_args(["--debug"]).debug is True


def test_add_out_models_dir_arg_is_required():
    parser = argparse.ArgumentParser()
    add_out_models_dir_arg(parser, help="where to write")
    assert parser.parse_args(["--out-models-dir", "x"]).out_models_dir == "x"
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_add_max_epochs_arg_requires_positive_int():
    parser = argparse.ArgumentParser()
    add_max_epochs_arg(parser)
    assert parser.parse_args([]).max_epochs is None
    assert parser.parse_args(["--max-epochs", "5"]).max_epochs == 5
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-epochs", "0"])
