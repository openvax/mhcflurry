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
