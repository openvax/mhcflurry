"""Tests for the small env-resolution helper in
``scripts/training/compare_new_vs_public.py``.

The script is not a package, so we load it via importlib rather than
importing through ``mhcflurry``.
"""
import importlib.util
import os
import pathlib


def _load_module():
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "scripts" / "training" / "compare_new_vs_public.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_new_vs_public", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_torchinductor_compile_threads_replaces_auto(monkeypatch):
    """Regression: TORCHINDUCTOR_COMPILE_THREADS=auto must be replaced
    with an integer in the parent env before the spawn-pool fires;
    PyTorch's decide_compile_threads() does int(os.environ[...]) on
    import and crashes on "auto", taking out every eval worker."""
    mod = _load_module()
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "auto")
    monkeypatch.delenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    monkeypatch.setenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_CAP", "16")
    mod._resolve_torchinductor_compile_threads(num_jobs=8)
    val = os.environ["TORCHINDUCTOR_COMPILE_THREADS"]
    assert val.isdigit(), f"expected integer, got {val!r}"
    assert int(val) >= 1
    assert os.environ["MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO"] == "1"


def test_resolve_torchinductor_compile_threads_skips_pinned(monkeypatch):
    """If the user pinned a numeric value, leave it alone."""
    mod = _load_module()
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "12")
    monkeypatch.delenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    mod._resolve_torchinductor_compile_threads(num_jobs=8)
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "12"
    assert "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO" not in os.environ


def test_resolve_torchinductor_compile_threads_no_op_when_unset(monkeypatch):
    """If the env var is unset, leave it unset; the worker's torch import
    will fall back to its own default."""
    mod = _load_module()
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    monkeypatch.delenv(
        "MHCFLURRY_TORCHINDUCTOR_COMPILE_THREADS_AUTO", raising=False)
    mod._resolve_torchinductor_compile_threads(num_jobs=8)
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ
