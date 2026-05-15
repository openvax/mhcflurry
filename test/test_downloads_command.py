"""Tests for ``mhcflurry.downloads_command``."""
import tarfile

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
