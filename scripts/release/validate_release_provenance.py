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

"""Validate and record source/model provenance for a release workflow."""

import argparse
import datetime
import json
import pathlib
import re
import subprocess


VERSION_RE = re.compile(r'^__version__\s*=\s*[\'\"]([^\'\"]+)[\'\"]', re.MULTILINE)
PACKAGE_RE = re.compile(r"^package\s+mhcflurry\s+(\S+)\s*$", re.MULTILINE)
BASE_VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)")


def base_version(version):
    """Return the three-component release version, excluding rc suffixes."""
    match = BASE_VERSION_RE.match(version)
    if not match:
        raise ValueError("Invalid MHCflurry version: %s" % version)
    return match.group(1)


def source_package_version(repo):
    """Read the package version without importing mhcflurry."""
    path = pathlib.Path(repo) / "mhcflurry" / "version.py"
    match = VERSION_RE.search(path.read_text())
    if not match:
        raise ValueError("Could not find __version__ in %s" % path)
    return match.group(1)


def model_package_version(path):
    """Read the training package version from a model info file."""
    match = PACKAGE_RE.search(pathlib.Path(path).read_text())
    if not match:
        raise ValueError("Could not find MHCflurry package version in %s" % path)
    return match.group(1)


def model_metadata(path):
    """Read the two-column model info file into normalized string fields."""
    result = {}
    for line in pathlib.Path(path).read_text().splitlines():
        fields = line.split("\t", 1)
        if len(fields) == 2:
            result[fields[0].strip()] = fields[1].strip()
    return result


def git_output(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo)] + list(args), text=True).strip()


def tracked_worktree_is_dirty(repo):
    """Return whether tracked files differ from HEAD (untracked files ignored)."""
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = subprocess.run(["git", "-C", str(repo)] + list(args))
        if result.returncode not in (0, 1):
            raise RuntimeError("git %s failed" % " ".join(args))
        if result.returncode == 1:
            return True
    return False


def artifact_info_paths(run_dir, processing_variants):
    """Return expected top-level model info paths keyed by artifact role."""
    run_dir = pathlib.Path(run_dir)
    result = {
        "affinity": run_dir / "affinity" / "models.combined" / "info.txt",
        "presentation": run_dir / "presentation" / "models" / "info.txt",
    }
    for variant in processing_variants:
        result["processing.%s" % variant] = (
            run_dir / "processing" / ("models.selected.%s" % variant) /
            "info.txt"
        )
    return result


def collect_provenance(
        repo, run_dir, release, workflow_id="", processing_variants=(),
        require_artifacts=False, allow_dirty_repo=False,
        expected_artifact_workflow_id=""):
    """Validate release identity and return serializable provenance."""
    repo = pathlib.Path(repo).resolve()
    run_dir = pathlib.Path(run_dir).resolve()
    source_version = source_package_version(repo)
    release_base = base_version(release)
    if base_version(source_version) != release_base:
        raise ValueError(
            "Release %s does not match source package version %s" % (
                release, source_version))

    dirty = tracked_worktree_is_dirty(repo)
    if dirty and not allow_dirty_repo:
        raise ValueError(
            "Tracked source files differ from HEAD; commit them before release "
            "or use --allow-dirty-repo for a deliberately non-release run")

    source_commit = git_output(repo, "rev-parse", "HEAD")
    artifact_provenance = {}
    missing = []
    for role, path in artifact_info_paths(
            run_dir, processing_variants).items():
        if path.is_file():
            version = model_package_version(path)
            if base_version(version) != release_base:
                raise ValueError(
                    "%s artifact was trained by MHCflurry %s, not release %s: "
                    "%s" % (role, version, release, path))
            metadata = model_metadata(path)
            artifact_commit = metadata.get("git commit", "")
            artifact_workflow_id = metadata.get("workflow id", "")
            if require_artifacts and artifact_commit != source_commit:
                raise ValueError(
                    "%s artifact git commit '%s' does not match source commit "
                    "'%s': %s" % (
                        role, artifact_commit or "missing", source_commit, path))
            if require_artifacts and expected_artifact_workflow_id and (
                    artifact_workflow_id != expected_artifact_workflow_id):
                raise ValueError(
                    "%s artifact workflow id '%s' does not match workflow "
                    "'%s': %s" % (
                        role, artifact_workflow_id or "missing",
                        expected_artifact_workflow_id, path))
            artifact_provenance[role] = {
                "package_version": version,
                "git_commit": artifact_commit,
                "workflow_id": artifact_workflow_id,
            }
        elif require_artifacts:
            missing.append(str(path))
    if missing:
        raise ValueError(
            "Missing required model provenance file(s): %s" %
            ", ".join(missing))

    return {
        "schema_version": 2,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "release": release,
        "release_base_version": release_base,
        "source": {
            "package_version": source_version,
            "git_commit": source_commit,
            "tracked_worktree_dirty": dirty,
        },
        "workflow_id": workflow_id,
        "artifacts": artifact_provenance,
    }


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--workflow-id", default="")
    parser.add_argument("--expected-artifact-workflow-id", default="")
    parser.add_argument(
        "--processing-variants",
        default="with_flanks no_flank short_flanks",
    )
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--allow-dirty-repo", action="store_true")
    parser.add_argument("--out")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        provenance = collect_provenance(
            repo=args.repo,
            run_dir=args.run_dir,
            release=args.release,
            workflow_id=args.workflow_id,
            processing_variants=args.processing_variants.split(),
            require_artifacts=args.require_artifacts,
            allow_dirty_repo=args.allow_dirty_repo,
            expected_artifact_workflow_id=args.expected_artifact_workflow_id,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit("ERROR: %s" % error) from error
    text = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name(out.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(out)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
