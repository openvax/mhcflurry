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

"""Tests for datacache-backed cache resolution, offline mode, dirs, and prune."""

from __future__ import annotations

import json
from pathlib import Path

import datacache.common as dc_common
import pytest

from mhcflurry import downloads
from mhcflurry.cli import downloads_command


@pytest.fixture(autouse=True)
def _restore_downloads_globals():
    """Preserve global mhcflurry.downloads state across tests."""
    prev_dir = downloads._DOWNLOADS_DIR
    prev_release = downloads._CURRENT_RELEASE
    yield
    downloads._DOWNLOADS_DIR = prev_dir
    downloads._CURRENT_RELEASE = prev_release


def test_resolve_data_location_precedence_and_legacy_preservation(monkeypatch, tmp_path: Path):
    legacy_dir = tmp_path / "legacy-v4"
    platform_shared = tmp_path / "platform-openvax"

    monkeypatch.delenv("MHCFLURRY_DOWNLOADS_DIR", raising=False)
    monkeypatch.delenv("MHCFLURRY_DATA_DIR", raising=False)
    monkeypatch.delenv("OPENVAX_DATA_CACHE", raising=False)
    monkeypatch.setattr(downloads, "user_data_dir", lambda *a, **kw: str(legacy_dir))
    platform_cache_attr = (
        "platform_cache_dir"
        if hasattr(dc_common, "platform_cache_dir")
        else "_platform_cache_dir"
    )
    monkeypatch.setattr(
        dc_common,
        platform_cache_attr,
        lambda app: str(platform_shared if app == "openvax" else tmp_path / app),
    )

    # Clean install defaults to shared platform root without creating it.
    loc = downloads.resolve_data_location()
    assert Path(loc.path) == platform_shared / "mhcflurry"
    assert loc.origin == "default"
    assert not Path(loc.path).exists()

    # Populated legacy platformdirs v4 directory is preserved (no forced re-download).
    (legacy_dir / "2.3.0").mkdir(parents=True)
    (legacy_dir / "2.3.0" / "marker.txt").write_text("existing", encoding="utf-8")
    loc_legacy = downloads.resolve_data_location()
    assert Path(loc_legacy.path) == legacy_dir
    assert loc_legacy.origin == "legacy"

    # OPENVAX_DATA_CACHE overrides legacy/default.
    shared_override = tmp_path / "openvax-shared"
    monkeypatch.setenv("OPENVAX_DATA_CACHE", str(shared_override))
    downloads.configure()
    loc_shared = downloads.resolve_data_location()
    assert Path(loc_shared.path) == shared_override / "mhcflurry"
    assert loc_shared.origin == "shared_env"
    assert Path(downloads.get_downloads_dir()).parent == shared_override / "mhcflurry"

    # MHCFLURRY_DATA_DIR takes highest precedence over OPENVAX_DATA_CACHE.
    pkg_override = tmp_path / "mhcflurry-override"
    monkeypatch.setenv("MHCFLURRY_DATA_DIR", str(pkg_override))
    downloads.configure()
    loc_pkg = downloads.resolve_data_location()
    assert Path(loc_pkg.path) == pkg_override
    assert loc_pkg.origin == "env"
    assert Path(downloads.get_downloads_dir()).parent == pkg_override


def test_dirs_subcommand_is_read_only_and_supports_json(monkeypatch, tmp_path: Path, capsys):
    shared = tmp_path / "shared"
    monkeypatch.delenv("MHCFLURRY_DOWNLOADS_DIR", raising=False)
    monkeypatch.delenv("MHCFLURRY_DATA_DIR", raising=False)
    monkeypatch.setenv("OPENVAX_DATA_CACHE", str(shared))
    monkeypatch.delenv("OPENVAX_OFFLINE", raising=False)
    monkeypatch.delenv("DATACACHE_OFFLINE", raising=False)
    downloads.configure()

    downloads_command.run(["dirs", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["package"] == "mhcflurry"
    assert payload["offline"] is False
    assert payload["data_root"]["path"] == str(shared / "mhcflurry")
    assert payload["data_root"]["origin"] == "shared_env"
    assert payload["data_root"]["exists"] is False
    assert not shared.exists(), "dirs must remain strictly read-only"

    downloads_command.run(["dirs"])
    text = capsys.readouterr().out
    assert "data_root" in text
    assert "active_downloads" in text
    assert not shared.exists()


def test_path_subcommand_json(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.delenv("MHCFLURRY_DOWNLOADS_DIR", raising=False)
    monkeypatch.setenv("MHCFLURRY_DATA_DIR", str(tmp_path))
    downloads.configure()

    downloads_command.run(["path", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["download"] is None
    assert payload["path"] == downloads.get_downloads_dir()


def test_prune_dry_run_and_apply(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.delenv("MHCFLURRY_DOWNLOADS_DIR", raising=False)
    monkeypatch.setenv("MHCFLURRY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MHCFLURRY_DOWNLOADS_CURRENT_RELEASE", "2.3.0")
    downloads.configure()

    current = tmp_path / "2.3.0" / "models_class1_presentation"
    stale = tmp_path / "2.2.0" / "models_class1_presentation"
    unrelated = tmp_path / "unrelated-dir"
    current.mkdir(parents=True)
    stale.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    (current / "weights.bin").write_bytes(b"1234")
    (stale / "weights.bin").write_bytes(b"12345678")
    (unrelated / "keep.txt").write_text("do not delete", encoding="utf-8")

    # Dry-run default deletes nothing.
    downloads_command.run(["prune", "--json"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert [e["release"] for e in plan["candidates"]] == ["2.2.0"]
    assert plan["deleted"] == []
    assert plan["reclaimable_bytes"] == 8
    assert stale.exists()
    assert unrelated.exists()

    # --yes deletes only stale release dirs, preserving current and unrelated dirs.
    downloads_command.run(["prune", "--yes", "--json"])
    applied = json.loads(capsys.readouterr().out)
    assert applied["dry_run"] is False
    assert [e["release"] for e in applied["deleted"]] == ["2.2.0"]
    assert applied["freed_bytes"] == 8
    assert not (tmp_path / "2.2.0").exists()
    assert current.exists()
    assert unrelated.exists()


def test_fetch_offline_blocks_network_before_mkdir(monkeypatch, tmp_path: Path, capsys):
    target_root = tmp_path / "offline-cache"
    monkeypatch.delenv("MHCFLURRY_DOWNLOADS_DIR", raising=False)
    monkeypatch.setenv("MHCFLURRY_DATA_DIR", str(target_root))
    downloads.configure()

    with pytest.raises(SystemExit) as excinfo:
        downloads_command.run(["fetch", "--offline", "models_class1_presentation"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "offline" in err.lower()
    assert not target_root.exists()
