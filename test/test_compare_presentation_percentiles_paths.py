"""Presentation-percentile provenance does not depend on the working directory."""

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/training/compare_presentation_percentiles.py"


def test_compact_transform_source_resolves_outside_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("compare_presentation_percentiles", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = module.COMPACT_TRANSFORM_SOURCE
    assert source.is_absolute() and source.is_file()
    assert source == REPO_ROOT / "mhcflurry/compact_percent_rank_transform.py"
    assert len(module.sha256(source)) == 64
