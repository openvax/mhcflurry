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

"""Tests for ``datacache``-backed ``mhcflurry.cli.downloads_command``."""
from argparse import Namespace
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from threading import Barrier

import datacache
import pytest

from mhcflurry import downloads
from mhcflurry.cli import downloads_command


@pytest.fixture(autouse=True)
def _restore_downloads_globals():
    """Preserve module-level download configuration across tests."""
    prev_dir = downloads._DOWNLOADS_DIR
    prev_release = downloads._CURRENT_RELEASE
    prev_meta = downloads._METADATA
    prev_models = downloads._MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR
    prev_pres = downloads._MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR
    prev_proc = downloads._MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR
    yield
    downloads._DOWNLOADS_DIR = prev_dir
    downloads._CURRENT_RELEASE = prev_release
    downloads._METADATA = prev_meta
    downloads._MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR = prev_models
    downloads._MHCFLURRY_DEFAULT_CLASS1_PRESENTATION_MODELS_DIR = prev_pres
    downloads._MHCFLURRY_DEFAULT_CLASS1_PROCESSING_MODELS_DIR = prev_proc


def _make_archive_bytes(files, *, unsafe_kind=None):
    archive_data = io.BytesIO()
    with tarfile.open(fileobj=archive_data, mode="w:bz2") as archive:
        if unsafe_kind == "traversal":
            info = tarfile.TarInfo("../escape.txt")
            data = b"escaped"
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        elif unsafe_kind == "symlink":
            info = tarfile.TarInfo("models/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../escape.txt"
            archive.addfile(info)
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return archive_data.getvalue()


def local_bundle(
    tmp_path,
    monkeypatch,
    *,
    multipart=False,
    unsafe_kind=None,
    pinned=False,
    bad_sha256=False,
    files=None,
    force=False,
    quiet=True,
):
    destination = tmp_path / "downloads"
    archives = tmp_path / "archives"
    archives.mkdir(exist_ok=True)
    if files is None:
        files = [("first.txt", b"first weights"), ("second.txt", b"second weights")]
    data = _make_archive_bytes(files, unsafe_kind=unsafe_kind)
    digest = hashlib.sha256(data).hexdigest()

    if multipart:
        split = len(data) // 2
        names = ["models.part1", "models.part2"]
        for name, part in zip(names, [data[:split], data[split:]]):
            (archives / name).write_bytes(part)
        source = {"part_urls": ["https://example.invalid/" + name for name in names]}
    else:
        (archives / "models.tar.bz2").write_bytes(data)
        source = {"url": "https://example.invalid/models.tar.bz2"}

    if pinned:
        source["sha256"] = ("0" * 64) if bad_sha256 else digest
        source["size"] = len(data)

    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", str(destination))
    monkeypatch.setattr(downloads, "_CURRENT_RELEASE", None)
    monkeypatch.setattr(downloads, "_METADATA", {
        "current-release": "test",
        "current-compatibility-version": 2,
        "bundles": {
            "models": {
                "kind": "models",
                "group": "Prediction models",
                "description": "Test models",
            },
        },
        "releases": {
            "test": {
                "compatibility-version": 2,
                "downloads": [dict(name="models", default=True, **source)],
            },
        },
    })
    args = Namespace(
        quiet=quiet,
        release="test",
        download_name=["models"],
        already_downloaded_dir=str(archives),
        keep=False,
        force=force,
        offline=False,
    )
    return args, destination, digest, len(data)


@pytest.mark.parametrize("failure", ["extract", "publish"])
def test_failed_install_is_absent_and_retryable(tmp_path, monkeypatch, failure):
    import datacache.archives as dc_archives

    args, destination, _digest, _size = local_bundle(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        if failure == "extract":
            def fail_extract(*a, **kw):
                raise OSError("injected extraction failure")

            patch.setattr(dc_archives, "_extract_tar", fail_extract)
        else:
            def fail_publish(*a, **kw):
                raise OSError("injected publication failure")

            patch.setattr(dc_archives.BundleStore, "publish", fail_publish)

        with pytest.raises(OSError, match="injected"):
            downloads_command.fetch_subcommand(args)

    assert not downloads.get_release_downloads("test")["models"]["downloaded"]
    with pytest.raises(RuntimeError, match="Missing MHCflurry downloadable file"):
        downloads.get_path("models", "first.txt", release="test")

    # Retry succeeds cleanly.
    downloads_command.fetch_subcommand(args)
    installed_root = Path(downloads.get_path("models", release="test"))
    assert (installed_root / "first.txt").read_bytes() == b"first weights"
    assert (installed_root / "second.txt").read_bytes() == b"second weights"
    assert downloads.get_release_downloads("test")["models"]["up_to_date"] is True


def test_multipart_archive_and_receipt_publish_together(tmp_path, monkeypatch):
    args, _destination, digest, size = local_bundle(
        tmp_path, monkeypatch, multipart=True
    )
    downloads_command.fetch_subcommand(args)
    installed = Path(downloads.get_path("models", release="test"))
    assert (installed / "second.txt").read_bytes() == b"second weights"
    receipt = (installed / "DOWNLOAD_INFO.csv").read_text(encoding="utf-8")
    assert "models.part1" in receipt
    assert "models.part2" in receipt

    info = downloads.get_release_downloads("test")["models"]
    assert info["downloaded"] is True
    assert info["up_to_date"] is True
    assert info["layout"] == "datacache"
    assert info["verified"] is False
    assert info["trusted_checksum"] is False
    assert info["recorded_sha256"] == digest
    assert info["archive_size"] == size
    assert info["fetched_at"] is not None


def test_pinned_checksum_passes_and_corrupt_archive_fails(tmp_path, monkeypatch, capsys):
    # Trusted checksum failure rejects installation without publishing.
    bad_args, _dest, _digest, _size = local_bundle(
        tmp_path, monkeypatch, pinned=True, bad_sha256=True
    )
    with pytest.raises(datacache.FileValidationError):
        downloads_command.fetch_subcommand(bad_args)
    assert not downloads.get_release_downloads("test")["models"]["downloaded"]

    # Valid trusted checksum installs cleanly and supports explicit --verify.
    good_args, _dest, digest, size = local_bundle(
        tmp_path, monkeypatch, pinned=True, bad_sha256=False
    )
    downloads_command.fetch_subcommand(good_args)

    # Fast offline inspection does not claim full verification.
    fast_info = downloads.get_release_downloads("test", verify_files=False)["models"]
    assert fast_info["downloaded"] is True
    assert fast_info["up_to_date"] is True
    assert fast_info["trusted_checksum"] is True
    assert fast_info["verified"] is False
    assert fast_info["recorded_sha256"] == digest
    assert fast_info["archive_size"] == size

    # Explicit full verification via CLI hashes every extracted file and sets verified=True.
    downloads_command.run(["info", "models", "--release", "test", "--verify", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["status"] == "installed; verified"
    assert payload["layout"] == "datacache"
    assert payload["recorded_sha256"] == digest
    assert payload["archive_size"] == size
    assert payload["fetched_at"] is not None


def test_unpinned_archive_never_reports_verified(tmp_path, monkeypatch, capsys):
    args, _dest, digest, _size = local_bundle(tmp_path, monkeypatch, pinned=False)
    downloads_command.fetch_subcommand(args)

    downloads_command.run(["info", "models", "--release", "test", "--verify", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["downloaded"] is True
    assert payload["source_matches"] is True
    assert payload["trusted_checksum"] is False
    assert payload["verified"] is False
    assert payload["recorded_sha256"] == digest


@pytest.mark.parametrize("unsafe_kind", ["traversal", "symlink"])
def test_unsafe_archive_never_publishes(tmp_path, monkeypatch, unsafe_kind):
    args, destination, _digest, _size = local_bundle(
        tmp_path, monkeypatch, unsafe_kind=unsafe_kind
    )
    with pytest.raises(datacache.FileValidationError):
        downloads_command.fetch_subcommand(args)
    assert not downloads.get_release_downloads("test")["models"]["downloaded"]
    assert not (destination.parent / "escape.txt").exists()


def test_existing_legacy_unreceipted_bundle_is_preserved(tmp_path, monkeypatch, capsys):
    args, destination, _digest, _size = local_bundle(tmp_path, monkeypatch)
    legacy_target = destination / "models"
    legacy_target.mkdir(parents=True)
    (legacy_target / "custom.txt").write_text("custom weights", encoding="utf-8")

    # Without --force, legacy directory is left completely untouched.
    downloads_command.fetch_subcommand(Namespace(**{**vars(args), "quiet": False}))
    assert [p.name for p in legacy_target.iterdir()] == ["custom.txt"]
    assert (legacy_target / "custom.txt").read_text(encoding="utf-8") == "custom weights"
    assert Path(downloads.get_path("models", release="test")) == legacy_target
    err = capsys.readouterr().err
    assert "--force" in err

    # With --force, installs into managed store while leaving legacy directory intact.
    force_args = Namespace(**{**vars(args), "force": True})
    downloads_command.fetch_subcommand(force_args)
    managed_root = Path(downloads.get_path("models", release="test"))
    assert managed_root != legacy_target
    assert (managed_root / "first.txt").read_bytes() == b"first weights"
    assert (legacy_target / "custom.txt").read_text(encoding="utf-8") == "custom weights"


def test_force_refresh_preserves_previous_generation_on_failure(tmp_path, monkeypatch):
    args, _dest, _digest, _size = local_bundle(
        tmp_path, monkeypatch, pinned=True, files=[("first.txt", b"gen1")]
    )
    downloads_command.fetch_subcommand(args)
    gen1_path = Path(downloads.get_path("models", release="test"))
    assert (gen1_path / "first.txt").read_bytes() == b"gen1"

    # Corrupt archive on --force fails without breaking existing generation.
    (Path(args.already_downloaded_dir) / "models.tar.bz2").write_bytes(b"corrupt")
    force_args = Namespace(**{**vars(args), "force": True})
    with pytest.raises(datacache.FileValidationError):
        downloads_command.fetch_subcommand(force_args)
    assert Path(downloads.get_path("models", release="test")) == gen1_path
    assert (gen1_path / "first.txt").read_bytes() == b"gen1"

    # Subsequent valid --force publishes a new generation while keeping gen1 readable.
    valid_bytes = _make_archive_bytes([("first.txt", b"gen2")])
    (Path(args.already_downloaded_dir) / "models.tar.bz2").write_bytes(valid_bytes)
    downloads._METADATA["releases"]["test"]["downloads"][0]["sha256"] = (
        hashlib.sha256(valid_bytes).hexdigest()
    )
    downloads._METADATA["releases"]["test"]["downloads"][0]["size"] = len(valid_bytes)
    downloads_command.fetch_subcommand(force_args)
    gen2_path = Path(downloads.get_path("models", release="test"))
    assert gen2_path != gen1_path
    assert (gen2_path / "first.txt").read_bytes() == b"gen2"
    assert (gen1_path / "first.txt").read_bytes() == b"gen1"


def test_concurrent_installers_publish_one_complete_bundle(tmp_path, monkeypatch):
    args, _destination, _digest, _size = local_bundle(tmp_path, monkeypatch)
    get_downloads = downloads_command.get_release_downloads
    barrier = Barrier(2)

    def simultaneous_selection(release):
        selected = get_downloads(release)
        barrier.wait(timeout=30)
        return selected

    monkeypatch.setattr(downloads_command, "get_release_downloads", simultaneous_selection)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(downloads_command.fetch_subcommand, args) for _ in range(2)]
        for result in futures:
            result.result(timeout=30)

    installed = Path(downloads.get_path("models", release="test"))
    assert (installed / "first.txt").read_bytes() == b"first weights"
    assert (installed / "second.txt").read_bytes() == b"second weights"
    assert (installed / "DOWNLOAD_INFO.csv").is_file()


def test_explicit_release_with_custom_downloads_dir(tmp_path, monkeypatch):
    """An explicit release must work when the custom-dir override is active."""
    release = "test-release"
    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", str(tmp_path))
    monkeypatch.setattr(downloads, "_CURRENT_RELEASE", None)
    monkeypatch.setattr(downloads, "_METADATA", {
        "releases": {
            release: {
                "downloads": [{
                    "name": "models",
                    "default": False,
                    "url": "https://example.invalid/models.tar.bz2",
                }],
            },
        },
    })

    available = downloads.get_release_downloads(release)
    assert list(available) == ["models"]
    assert not available["models"]["downloaded"]


def test_fetch_uses_explicit_release(tmp_path, monkeypatch):
    seen_releases = []

    def fake_downloads(release):
        seen_releases.append(release)
        return OrderedDict()

    monkeypatch.setattr(downloads_command, "get_release_downloads", fake_downloads)
    downloads_command.fetch_subcommand(Namespace(
        quiet=True,
        release="test-release",
        download_name=[],
        already_downloaded_dir=None,
        keep=False,
        force=False,
        offline=False,
    ))
    assert seen_releases == ["test-release"]


@pytest.mark.parametrize("custom_dir", [False, True])
def test_fetch_requested_release_uses_its_own_status_and_destination(
        tmp_path, monkeypatch, capsys, custom_dir):
    current_dir = tmp_path / "current"
    (current_dir / "models").mkdir(parents=True)
    (current_dir / "models" / "sentinel").write_text("current weights", encoding="utf-8")
    destination = tmp_path / ("custom" if custom_dir else "older")
    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", str(
        destination if custom_dir else current_dir))
    monkeypatch.setattr(downloads, "_CURRENT_RELEASE", None if custom_dir else "current")
    monkeypatch.setattr(downloads, "_METADATA", {"releases": {"older": {
        "downloads": [{"name": "models", "default": True,
                       "url": "https://example.invalid/models.tar.bz2"}],
    }}})
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "models.tar.bz2").write_bytes(
        _make_archive_bytes([("weights.txt", b"older weights")])
    )
    args = Namespace(
        quiet=True,
        release="older",
        download_name=["models"],
        already_downloaded_dir=str(archives),
        keep=False,
        force=False,
        offline=False,
    )
    assert not downloads.get_release_downloads("older")["models"]["downloaded"]
    downloads_command.fetch_subcommand(args)
    installed = Path(downloads.get_path("models", release="older"))
    assert installed.is_relative_to(destination)
    assert (installed / "weights.txt").read_text(encoding="utf-8") == "older weights"
    assert (current_dir / "models/sentinel").read_text(encoding="utf-8") == "current weights"
    assert not (current_dir / "models/weights.txt").exists()
    assert downloads.get_release_downloads("older")["models"]["up_to_date"] is True

    # --quiet suppresses the already-installed banner (#428); verbose/default
    # writes it to stderr, leaving stdout clean.
    downloads_command.fetch_subcommand(args)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    downloads_command.fetch_subcommand(Namespace(**{**vars(args), "quiet": False}))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(installed) in captured.err


def test_fetch_status_writes_to_stderr_and_relative_override_finds_managed_bundle(
        tmp_path, monkeypatch, capsys):
    args, _dest, _digest, _size = local_bundle(
        tmp_path,
        monkeypatch,
        quiet=False,
        files=[("models.combined/manifest.csv", b"model_name\n")],
    )
    downloads_command.fetch_subcommand(args)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Fetching 1/1 downloads" in captured.err

    # Relative MHCFLURRY_DEFAULT_CLASS1_MODELS resolves through managed store.
    monkeypatch.setattr(
        downloads, "_MHCFLURRY_DEFAULT_CLASS1_MODELS_DIR", "models/models.combined"
    )
    resolved = Path(downloads.get_default_class1_models_dir(test_exists=True))
    assert (resolved / "manifest.csv").is_file()


def test_corrupt_managed_store_requires_force_to_repair(tmp_path, monkeypatch):
    args, _dest, _digest, _size = local_bundle(tmp_path, monkeypatch, pinned=True)
    downloads_command.fetch_subcommand(args)

    # Corrupt the published bundle manifest so the managed store becomes invalid.
    installed = Path(downloads.get_path("models", release="test"))
    (installed / ".datacache-manifest.json").write_text("{broken json", encoding="utf-8")

    info = downloads.get_release_downloads("test")["models"]
    assert info["downloaded"] is False
    assert info["install_status"] == "invalid"

    # Without --force, datacache refuses to silently overwrite an invalid store.
    with pytest.raises(datacache.FileValidationError):
        downloads_command.fetch_subcommand(args)

    # With --force, repairs cleanly by publishing a new valid generation.
    force_args = Namespace(**{**vars(args), "force": True})
    downloads_command.fetch_subcommand(force_args)
    repaired = Path(downloads.get_path("models", release="test"))
    assert (repaired / "first.txt").read_bytes() == b"first weights"
    assert downloads.get_release_downloads("test", verify_files=True)["models"]["verified"] is True


def test_custom_root_managed_store_enforces_release_source_match(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_CURRENT_RELEASE", None)
    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", str(tmp_path))
    archives = tmp_path / "archives"
    archives.mkdir()
    entry_230 = downloads.get_release_downloads("2.3.0")["models_class1_presentation"]["metadata"]
    archive_name = Path(entry_230["url"]).name
    data = _make_archive_bytes([("models/weights.txt", b"2.3.0 weights")])
    (archives / archive_name).write_bytes(data)

    # Patch only the 2.3.0 presentation checksum/size to match our tiny test archive.
    meta_copy = json.loads(json.dumps(downloads.get_downloads_metadata()))
    for item in meta_copy["releases"]["2.3.0"]["downloads"]:
        if item["name"] == "models_class1_presentation":
            item["sha256"] = hashlib.sha256(data).hexdigest()
            item["size"] = len(data)
    monkeypatch.setattr(downloads, "_METADATA", meta_copy)

    downloads_command.fetch_subcommand(Namespace(
        quiet=True,
        release="2.3.0",
        download_name=["models_class1_presentation"],
        already_downloaded_dir=str(archives),
        keep=False,
        force=False,
        offline=False,
    ))
    resolved_230 = Path(downloads.get_model_release_dir("2.3.0"))
    assert (resolved_230 / "weights.txt").read_bytes() == b"2.3.0 weights"
    # Must refuse to serve the 2.3.0 managed bundle when release 2.2.0 is requested.
    with pytest.raises(ValueError, match="Cannot confirm"):
        downloads.get_model_release_dir("2.2.0")


def test_model_loading_smoke_from_datacache_installed_archive(tmp_path, monkeypatch):
    import numpy
    from mhcflurry import Class1AffinityPredictor, Class1NeuralNetwork

    net = Class1NeuralNetwork(layer_sizes=[4], max_epochs=1, validation_split=0.0)
    net.fit(peptides=["SIINFEKL", "NLVPMVATV"], affinities=[50.0, 5000.0], verbose=0)
    predictor = Class1AffinityPredictor(
        allele_to_allele_specific_models={"HLA-A*02:01": [net]}
    )
    saved_dir = tmp_path / "src_models"
    predictor.save(str(saved_dir))
    expected_pred = predictor.predict(["SIINFEKL", "NLVPMVATV"], allele="HLA-A*02:01")

    # Pack saved model into a tar.bz2 and install via datacache-backed fetch.
    files = [
        ("models/" + str(p.relative_to(saved_dir)), p.read_bytes())
        for p in sorted(saved_dir.rglob("*"))
        if p.is_file()
    ]
    args, _dest, _digest, _size = local_bundle(
        tmp_path, monkeypatch, pinned=True, files=files
    )
    downloads_command.fetch_subcommand(args)
    loaded = Class1AffinityPredictor.load(
        downloads.get_path("models", "models", release="test")
    )
    actual_pred = loaded.predict(["SIINFEKL", "NLVPMVATV"], allele="HLA-A*02:01")
    numpy.testing.assert_allclose(actual_pred, expected_pred, rtol=1e-6, atol=0.0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only allele filenames with * and :")
def test_legacy_posix_allele_filenames_install_and_verify(tmp_path, monkeypatch):
    member = "models/weights_HLA-A*02:06-1-50c5f97331dc8241.npz"
    args, _dest, _digest, _size = local_bundle(
        tmp_path,
        monkeypatch,
        pinned=False,
        files=[(member, b"legacy-npz-bytes"), ("manifest.csv", b"model_name\n")],
    )
    downloads_command.fetch_subcommand(args)
    installed = Path(downloads.get_path("models", release="test"))
    assert (installed / member).read_bytes() == b"legacy-npz-bytes"
    info = downloads.get_release_downloads("test", verify_files=True)["models"]
    assert info["downloaded"] is True
    assert info["up_to_date"] is True
    assert info["install_status"] == "available"


def test_install_download_forwards_default_timeout_and_retries(tmp_path, monkeypatch):
    """Regression test for #462: bound hung downloads with explicit timeout and retries."""
    seen = []

    def fake_install_archive(store_dir, sources, **kwargs):
        seen.append(kwargs["download_options"])
        return Path(store_dir) / "bundles" / "fake"

    monkeypatch.setattr(datacache, "install_archive", fake_install_archive)
    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", str(tmp_path))
    monkeypatch.setattr(downloads, "_CURRENT_RELEASE", "test")
    entry = {
        "name": "models",
        "url": "https://example.invalid/models.tar.bz2",
    }

    downloads.install_download("models", release="test", metadata=entry)
    assert seen[-1]["timeout"] == downloads.DEFAULT_DOWNLOAD_TIMEOUT
    assert seen[-1]["max_retries"] == downloads.DEFAULT_DOWNLOAD_MAX_RETRIES

    downloads.install_download(
        "models", release="test", metadata=entry, timeout=15, max_retries=1
    )
    assert seen[-1]["timeout"] == 15
    assert seen[-1]["max_retries"] == 1



