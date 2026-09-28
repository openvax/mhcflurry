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

"""Tests for ``mhcflurry.downloads_command``."""
from argparse import Namespace
from collections import OrderedDict
import io
import tarfile
import pytest

from mhcflurry import downloads
from mhcflurry.cli import downloads_command
from mhcflurry.downloads_command import suspicious_tar_member


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
