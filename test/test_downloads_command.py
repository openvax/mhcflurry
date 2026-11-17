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

"""Tests for ``mhcflurry.cli.downloads_command``."""
from argparse import Namespace
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import tarfile
from threading import Barrier
import pytest

from mhcflurry import downloads
from mhcflurry.cli import downloads_command
from mhcflurry.cli.downloads_command import suspicious_tar_member


def _tar_info(name, type_=tarfile.REGTYPE, linkname=""):
    info = tarfile.TarInfo(name)
    info.type = type_
    info.linkname = linkname
    return info


def test_suspicious_tar_member_rejects_path_traversal_and_links():
    assert not suspicious_tar_member(_tar_info("models/file.txt"))

    assert suspicious_tar_member(_tar_info("/tmp/file.txt"))
    assert suspicious_tar_member(_tar_info("../file.txt"))
    assert suspicious_tar_member(_tar_info("models/.."))
    assert suspicious_tar_member(_tar_info("models/../../file.txt"))
    assert suspicious_tar_member(_tar_info("models/link", tarfile.SYMTYPE, "../x"))
    assert suspicious_tar_member(_tar_info("models/hard", tarfile.LNKTYPE, "../x"))
    for type_ in [tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE]:
        assert suspicious_tar_member(_tar_info("models/special", type_))


def local_bundle(tmp_path, monkeypatch, multipart=False, unsafe=False):
    destination = tmp_path / "downloads"
    archives = tmp_path / "archives"
    archives.mkdir()
    archive_data = io.BytesIO()
    with tarfile.open(fileobj=archive_data, mode="w:bz2") as archive:
        for name, data in [("first.txt", b"first weights"), ("second.txt", b"second weights")]:
            info = tarfile.TarInfo("../escape.txt" if unsafe else name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    data = archive_data.getvalue()
    if multipart:
        split = len(data) // 2
        names = ["models.part1", "models.part2"]
        for name, part in zip(names, [data[:split], data[split:]]):
            (archives / name).write_bytes(part)
        source = {"part_urls": ["https://example.invalid/" + name for name in names]}
    else:
        (archives / "models.tar.bz2").write_bytes(data)
        source = {"url": "https://example.invalid/models.tar.bz2"}
    monkeypatch.setattr(downloads, "_DOWNLOADS_DIR", str(destination))
    monkeypatch.setattr(downloads, "_CURRENT_RELEASE", None)
    monkeypatch.setattr(downloads, "_METADATA", {"releases": {"test": {
        "downloads": [dict(name="models", default=True, **source)],
    }}})
    args = Namespace(quiet=True, release="test", download_name=["models"],
                     already_downloaded_dir=str(archives), keep=False)
    return args, destination / "models"


@pytest.mark.parametrize("failure", ["extract", "receipt", "publish"])
def test_failed_install_is_absent_and_retryable(tmp_path, monkeypatch, failure):
    args, target = local_bundle(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        if failure == "extract":
            original = tarfile.TarFile.extractall
            calls = []

            def extract(archive, *a, **kw):
                calls.append(1)
                if len(calls) == 2:
                    raise OSError("injected extraction failure")
                return original(archive, *a, **kw)

            patch.setattr(tarfile.TarFile, "extractall", extract)
        elif failure == "receipt":
            import builtins

            def open_receipt(path, *a, **kw):
                if str(path).endswith("DOWNLOAD_INFO.csv"):
                    raise OSError("injected receipt failure")
                return builtins.open(path, *a, **kw)

            patch.setattr(downloads_command, "open", open_receipt, raising=False)
        else:
            def rename(*a, **kw):
                raise OSError("injected publication failure")

            patch.setattr(downloads_command.os, "rename", rename)
        with pytest.raises(OSError, match="injected"):
            downloads_command.fetch_subcommand(args)
    assert not target.exists()
    assert not list(target.parent.glob(".models-stage-*"))
    assert not downloads.get_release_downloads("test")["models"]["downloaded"]
    downloads_command.fetch_subcommand(args)
    assert (target / "first.txt").read_bytes() == b"first weights"
    assert (target / "second.txt").read_bytes() == b"second weights"
    assert downloads.get_release_downloads("test")["models"]["up_to_date"]


def test_multipart_archive_and_receipt_publish_together(tmp_path, monkeypatch):
    args, target = local_bundle(tmp_path, monkeypatch, multipart=True)
    downloads_command.fetch_subcommand(args)
    assert (target / "second.txt").read_bytes() == b"second weights"
    assert "models.part1" in (target / "DOWNLOAD_INFO.csv").read_text()
    assert "models.part2" in (target / "DOWNLOAD_INFO.csv").read_text()
    assert downloads.get_release_downloads("test")["models"]["up_to_date"]


def test_suspicious_archive_never_publishes(tmp_path, monkeypatch):
    args, target = local_bundle(tmp_path, monkeypatch, unsafe=True)
    with pytest.raises(RuntimeError, match="suspicious"):
        downloads_command.fetch_subcommand(args)
    assert not target.exists()
    assert not (target.parent.parent / "escape.txt").exists()


def test_existing_unreceipted_bundle_is_preserved(tmp_path, monkeypatch):
    args, target = local_bundle(tmp_path, monkeypatch)
    target.mkdir(parents=True)
    (target / "custom.txt").write_text("custom weights")
    downloads_command.fetch_subcommand(args)
    assert [p.name for p in target.iterdir()] == ["custom.txt"]
    assert (target / "custom.txt").read_text() == "custom weights"


def test_concurrent_installers_publish_one_complete_bundle(tmp_path, monkeypatch):
    args, target = local_bundle(tmp_path, monkeypatch)
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
    assert (target / "first.txt").read_bytes() == b"first weights"
    assert (target / "second.txt").read_bytes() == b"second weights"
    assert (target / "DOWNLOAD_INFO.csv").is_file()
    assert not list(target.parent.glob(".models-stage-*"))


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

    monkeypatch.setattr(
        downloads_command, "get_release_downloads", fake_downloads)
    monkeypatch.setattr(
        downloads_command, "get_downloads_dir", lambda release=None: str(tmp_path))

    downloads_command.fetch_subcommand(Namespace(
        quiet=True,
        release="test-release",
        download_name=[],
        already_downloaded_dir=None,
        keep=False,
    ))

    assert seen_releases == ["test-release"]


@pytest.mark.parametrize("custom_dir", [False, True])
def test_fetch_requested_release_uses_its_own_status_and_destination(
        tmp_path, monkeypatch, capsys, custom_dir):
    current_dir = tmp_path / "current"
    (current_dir / "models").mkdir(parents=True)
    (current_dir / "models" / "sentinel").write_text("current weights")
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
    with tarfile.open(archives / "models.tar.bz2", "w:bz2") as archive:
        member = tarfile.TarInfo("weights.txt")
        member.size = len(b"older weights")
        archive.addfile(member, io.BytesIO(b"older weights"))
    args = Namespace(quiet=True, release="older", download_name=["models"],
                     already_downloaded_dir=str(archives), keep=False)
    assert not downloads.get_release_downloads("older")["models"]["downloaded"]
    downloads_command.fetch_subcommand(args)
    assert (destination / "models/weights.txt").read_text() == "older weights"
    assert (current_dir / "models/sentinel").read_text() == "current weights"
    assert not (current_dir / "models/weights.txt").exists()
    assert downloads.get_release_downloads("older")["models"]["up_to_date"]
    downloads_command.fetch_subcommand(args)
    assert str(destination / "models") in capsys.readouterr().out


def test_download_progress_hook_reports_bytes_transferred():
    class Bar:
        n = 0
        total = None

        def update(self, increment):
            self.n += increment

    bar = Bar()
    hook = downloads_command.download_progress_hook(bar)
    hook(0, 8192, 20000)
    assert (bar.n, bar.total) == (0, 20000)
    hook(2, 8192, 20000)
    assert bar.n == 16384
    hook(3, 8192)
    assert (bar.n, bar.total) == (24576, 20000)
