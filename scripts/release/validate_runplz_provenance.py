#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the runplz installation selected by the release workflow."""

import argparse
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import urllib.parse
import urllib.request


def editable_install_root(distribution):
    """Return the PEP 610 editable source root, if this is an editable install."""
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return None
    direct_url = json.loads(direct_url_text)
    if not direct_url.get("dir_info", {}).get("editable", False):
        return None
    parsed = urllib.parse.urlsplit(direct_url.get("url", ""))
    if parsed.scheme != "file":
        raise ValueError(
            "editable runplz install has a non-file source URL: %s" % (
                direct_url.get("url"),
            )
        )
    return pathlib.Path(
        urllib.request.url2pathname(parsed.path)
    ).resolve()


def validate_runplz_provenance(executable, required_version):
    """Validate version and checkout cleanliness for the active interpreter."""
    try:
        distribution = importlib.metadata.distribution("runplz")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            "runplz package metadata was not found in selected interpreter %s" % (
                sys.executable,
            )
        ) from error
    actual_version = distribution.version
    if actual_version != required_version:
        raise ValueError(
            "runplz version mismatch: required %s, found %s" % (
                required_version,
                actual_version,
            )
        )

    import runplz

    executable = pathlib.Path(executable).resolve()
    module_path = pathlib.Path(runplz.__file__).resolve()
    identity = "executable=%s interpreter=%s version=%s module=%s" % (
        executable,
        pathlib.Path(sys.executable).resolve(),
        actual_version,
        module_path,
    )
    editable_root = editable_install_root(distribution)
    if editable_root is not None:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(editable_root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(editable_root), "status", "--porcelain"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError as error:
            raise ValueError(
                "editable runplz install is not a usable git checkout: %s" % (
                    editable_root,
                )
            ) from error
        if dirty:
            raise ValueError(
                "editable runplz checkout is dirty: %s" % editable_root
            )
        identity += " editable_root=%s git_commit=%s" % (
            editable_root,
            commit,
        )
    return identity


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--required-version", required=True)
    args = parser.parse_args(argv)
    try:
        identity = validate_runplz_provenance(
            args.executable,
            args.required_version,
        )
    except (ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print("runplz provenance: %s" % identity)


if __name__ == "__main__":
    main()
