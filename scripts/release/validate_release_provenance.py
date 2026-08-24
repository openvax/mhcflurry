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
import hashlib
import json
import pathlib
import re
import subprocess


VERSION_RE = re.compile(r'^__version__\s*=\s*[\'\"]([^\'\"]+)[\'\"]', re.MULTILINE)
PACKAGE_RE = re.compile(r"^package\s+mhcflurry\s+(\S+)\s*$", re.MULTILINE)
BASE_VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)")
PROCESSING_VARIANTS = ("no_flank", "with_flanks", "short_flanks")
HOLDOUT_MANIFESTS = (
    "affinity_pmhcs.csv",
    "affinity_samples.csv",
    "processing_samples.csv",
    "presentation_samples.csv",
)


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


def sha256(path):
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_holdout_provenance(run_dir, require_artifacts=False):
    """Validate and summarize the release holdout proof files."""
    holdout_dir = pathlib.Path(run_dir) / "release_holdout"
    policy_path = holdout_dir / "policy.json"
    validation_path = holdout_dir / "validation.json"
    if not policy_path.is_file() or not validation_path.is_file():
        if require_artifacts:
            raise ValueError(
                "Missing required release holdout policy or validation: %s" %
                holdout_dir)
        return {}

    policy = json.loads(policy_path.read_text())
    validation = json.loads(validation_path.read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported release holdout policy schema")
    if validation.get("schema_version") != 1:
        raise ValueError("Unsupported release holdout validation schema")
    if validation.get("policy_sha256") != sha256(policy_path):
        raise ValueError(
            "Release holdout validation does not match policy.json")

    manifest_records = policy.get("holdout_files", {})
    if set(manifest_records) != set(HOLDOUT_MANIFESTS):
        raise ValueError(
            "Release holdout policy has unexpected manifests: %s" %
            sorted(manifest_records))
    if validation.get("holdout_files") != manifest_records:
        raise ValueError(
            "Release holdout validation manifest records do not match policy")
    for filename in HOLDOUT_MANIFESTS:
        path = holdout_dir / filename
        expected = manifest_records[filename]
        if not path.is_file() or sha256(path) != expected.get("sha256"):
            raise ValueError(
                "Release holdout manifest checksum mismatch: %s" % path)

    overlaps = {
        key: validation.get(key)
        for key in (
            "affinity_overlap_rows",
            "processing_overlap_rows",
            "presentation_overlap_rows",
        )
    }
    if any(value != 0 for value in overlaps.values()):
        raise ValueError(
            "Release holdout validation contains overlap: %s" % overlaps)
    return {
        "policy_sha256": sha256(policy_path),
        "validation_sha256": sha256(validation_path),
        "evaluation_row_counts": policy.get("evaluation_row_counts", {}),
        "evaluation_hit_counts": policy.get("evaluation_hit_counts", {}),
        "presentation_holdout_pmids": policy.get(
            "presentation_holdout_pmids", []),
        "holdout_files": manifest_records,
        "overlap_rows": overlaps,
    }


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
    processing_variants = tuple(processing_variants)
    invalid = sorted(set(processing_variants) - set(PROCESSING_VARIANTS))
    if invalid:
        raise ValueError(
            "Unknown processing variant(s): %s. Expected: %s" % (
                ", ".join(invalid), ", ".join(PROCESSING_VARIANTS))
        )
    duplicates = sorted({
        value for value in processing_variants
        if processing_variants.count(value) > 1
    })
    if duplicates:
        raise ValueError(
            "Duplicate processing variant(s): %s" % ", ".join(duplicates)
        )
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


def collect_artifact_provenance(
        run_dir, release, processing_variants=(), require_artifacts=False,
        expected_artifact_git_commit="",
        expected_artifact_workflow_id=""):
    """Validate model metadata without requiring a Git checkout."""
    run_dir = pathlib.Path(run_dir).resolve()
    release_base = base_version(release)
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
            if require_artifacts and expected_artifact_git_commit and (
                    artifact_commit != expected_artifact_git_commit):
                raise ValueError(
                    "%s artifact git commit '%s' does not match source commit "
                    "'%s': %s" % (
                        role, artifact_commit or "missing",
                        expected_artifact_git_commit, path))
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
    if require_artifacts:
        missing_commits = sorted(
            role for role, item in artifact_provenance.items()
            if not item["git_commit"])
        if missing_commits:
            raise ValueError(
                "Model artifact provenance is missing git commit for: %s" %
                ", ".join(missing_commits))
        commits = {
            item["git_commit"] for item in artifact_provenance.values()
        }
        if len(commits) > 1:
            raise ValueError(
                "Model artifacts do not agree on git commit: %s" %
                ", ".join(sorted(commits)))
        missing_workflows = sorted(
            role for role, item in artifact_provenance.items()
            if not item["workflow_id"])
        if missing_workflows:
            raise ValueError(
                "Model artifact provenance is missing workflow id for: %s" %
                ", ".join(missing_workflows))
        workflow_ids = {
            item["workflow_id"] for item in artifact_provenance.values()
        }
        if len(workflow_ids) > 1:
            raise ValueError(
                "Model artifacts do not agree on workflow id: %s" %
                ", ".join(sorted(workflow_ids)))
    return artifact_provenance


def collect_provenance(
        repo, run_dir, release, workflow_id="", processing_variants=(),
        require_artifacts=False, allow_dirty_repo=False,
        expected_artifact_workflow_id="",
        allow_artifact_source_mismatch=False):
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
    artifact_provenance = collect_artifact_provenance(
        run_dir=run_dir,
        release=release,
        processing_variants=processing_variants,
        require_artifacts=require_artifacts,
        expected_artifact_git_commit=(
            source_commit
            if require_artifacts and not allow_artifact_source_mismatch
            else ""),
        expected_artifact_workflow_id=expected_artifact_workflow_id,
    )
    artifact_commits = {
        item["git_commit"] for item in artifact_provenance.values()
        if item["git_commit"]
    }
    artifact_workflow_ids = {
        item["workflow_id"] for item in artifact_provenance.values()
        if item["workflow_id"]
    }
    artifact_training = {}
    if artifact_commits:
        artifact_training = {
            "git_commit": next(iter(artifact_commits)),
            "workflow_id": next(iter(artifact_workflow_ids), ""),
            "matches_evaluation_source": artifact_commits == {source_commit},
        }
    holdout_provenance = collect_holdout_provenance(
        run_dir, require_artifacts=require_artifacts)

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
        "artifact_training": artifact_training,
        "artifacts": artifact_provenance,
        "release_holdout": holdout_provenance,
    }


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--workflow-id", default="")
    parser.add_argument("--artifact-only", action="store_true")
    parser.add_argument("--expected-artifact-git-commit", default="")
    parser.add_argument("--expected-artifact-workflow-id", default="")
    parser.add_argument(
        "--processing-variants",
        default="with_flanks no_flank short_flanks",
    )
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--allow-dirty-repo", action="store_true")
    parser.add_argument(
        "--allow-artifact-source-mismatch",
        action="store_true",
        help=(
            "Permit a postprocess/deploy checkout newer than the model-training "
            "commit while still requiring all model artifacts to agree on "
            "their recorded commit and workflow."
        ),
    )
    parser.add_argument("--out")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.artifact_only:
            if args.repo:
                raise ValueError(
                    "--repo is not used with --artifact-only; omit it")
            if not args.require_artifacts:
                raise ValueError(
                    "--artifact-only requires --require-artifacts")
            if not args.expected_artifact_git_commit:
                raise ValueError(
                    "--artifact-only requires --expected-artifact-git-commit")
            if not args.expected_artifact_workflow_id:
                raise ValueError(
                    "--artifact-only requires "
                    "--expected-artifact-workflow-id")
            artifacts = collect_artifact_provenance(
                run_dir=args.run_dir,
                release=args.release,
                processing_variants=args.processing_variants.split(),
                require_artifacts=True,
                expected_artifact_git_commit=(
                    args.expected_artifact_git_commit),
                expected_artifact_workflow_id=(
                    args.expected_artifact_workflow_id),
            )
            holdout = collect_holdout_provenance(
                args.run_dir, require_artifacts=True)
            provenance = {
                "schema_version": 2,
                "recorded_at": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                "release": args.release,
                "release_base_version": base_version(args.release),
                "source": {
                    "expected_git_commit": (
                        args.expected_artifact_git_commit),
                },
                "workflow_id": args.workflow_id,
                "artifacts": artifacts,
                "release_holdout": holdout,
            }
        else:
            if not args.repo:
                raise ValueError("--repo is required unless --artifact-only")
            provenance = collect_provenance(
                repo=args.repo,
                run_dir=args.run_dir,
                release=args.release,
                workflow_id=args.workflow_id,
                processing_variants=args.processing_variants.split(),
                require_artifacts=args.require_artifacts,
                allow_dirty_repo=args.allow_dirty_repo,
                expected_artifact_workflow_id=(
                    args.expected_artifact_workflow_id),
                allow_artifact_source_mismatch=(
                    args.allow_artifact_source_mismatch),
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
