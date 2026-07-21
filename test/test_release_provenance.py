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

import importlib.util
import pathlib
import subprocess

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "release" / "validate_release_provenance.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_release_provenance_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_model_info(run_dir, version, commit=None, workflow_id="run-123"):
    if commit is None:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    paths = [
        run_dir / "affinity" / "models.combined" / "info.txt",
        run_dir / "presentation" / "models" / "info.txt",
        run_dir / "processing" / "models.selected.with_flanks" / "info.txt",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "package\tmhcflurry %s\n"
            "git commit\t%s\n"
            "workflow id\t%s\n" % (version, commit, workflow_id)
        )


def test_collect_provenance_accepts_matching_release_candidate(tmp_path):
    module = load_module()
    write_model_info(tmp_path, "2.3.0rc14")

    result = module.collect_provenance(
        repo=REPO,
        run_dir=tmp_path,
        release="2.3.0",
        workflow_id="run-123",
        processing_variants=["with_flanks"],
        require_artifacts=True,
        allow_dirty_repo=True,
        expected_artifact_workflow_id="run-123",
    )

    assert result["release_base_version"] == "2.3.0"
    assert result["source"]["package_version"] == "2.3.0rc14"
    assert result["workflow_id"] == "run-123"
    assert {
        item["package_version"] for item in result["artifacts"].values()
    } == {"2.3.0rc14"}


def test_collect_provenance_rejects_mislabeled_model(tmp_path):
    module = load_module()
    write_model_info(tmp_path, "2.3.1")

    with pytest.raises(ValueError, match="2.3.1, not release 2.3.0"):
        module.collect_provenance(
            repo=REPO,
            run_dir=tmp_path,
            release="2.3.0",
            processing_variants=["with_flanks"],
            require_artifacts=True,
            allow_dirty_repo=True,
            expected_artifact_workflow_id="run-123",
        )


def test_collect_provenance_rejects_missing_model_info(tmp_path):
    module = load_module()

    with pytest.raises(ValueError, match="Missing required model provenance"):
        module.collect_provenance(
            repo=REPO,
            run_dir=tmp_path,
            release="2.3.0",
            processing_variants=["with_flanks"],
            require_artifacts=True,
            allow_dirty_repo=True,
            expected_artifact_workflow_id="run-123",
        )


@pytest.mark.parametrize(
    ("variants", "message"),
    [
        (["with_flanks", "../../outside"], "Unknown processing variant"),
        (["with_flanks", "with_flanks"], "Duplicate processing variant"),
    ],
)
def test_artifact_paths_reject_invalid_processing_variants(
        tmp_path, variants, message):
    module = load_module()
    with pytest.raises(ValueError, match=message):
        module.artifact_info_paths(tmp_path, variants)


def test_collect_provenance_rejects_model_from_another_commit(tmp_path):
    module = load_module()
    write_model_info(tmp_path, "2.3.0rc14", commit="deadbeef")

    with pytest.raises(ValueError, match="does not match source commit"):
        module.collect_provenance(
            repo=REPO,
            run_dir=tmp_path,
            release="2.3.0",
            workflow_id="run-123",
            processing_variants=["with_flanks"],
            require_artifacts=True,
            allow_dirty_repo=True,
            expected_artifact_workflow_id="run-123",
        )


def test_collect_provenance_allows_later_evaluation_workflow(tmp_path):
    module = load_module()
    write_model_info(tmp_path, "2.3.0rc14", workflow_id="training-run")

    result = module.collect_provenance(
        repo=REPO,
        run_dir=tmp_path,
        release="2.3.0",
        workflow_id="evaluation-run",
        processing_variants=["with_flanks"],
        require_artifacts=True,
        allow_dirty_repo=True,
    )

    assert result["workflow_id"] == "evaluation-run"
    assert result["artifacts"]["affinity"]["workflow_id"] == "training-run"


def test_collect_artifact_provenance_without_git_checkout(tmp_path):
    module = load_module()
    write_model_info(
        tmp_path,
        "2.3.0rc14",
        commit="remote-commit",
        workflow_id="remote-run",
    )

    result = module.collect_artifact_provenance(
        run_dir=tmp_path,
        release="2.3.0",
        processing_variants=["with_flanks"],
        require_artifacts=True,
        expected_artifact_git_commit="remote-commit",
        expected_artifact_workflow_id="remote-run",
    )

    assert result["affinity"]["git_commit"] == "remote-commit"


def test_artifact_only_cli_requires_expected_identity(tmp_path):
    module = load_module()
    write_model_info(tmp_path, "2.3.0rc14")

    with pytest.raises(SystemExit, match="expected-artifact-git-commit"):
        module.main([
            "--artifact-only",
            "--run-dir", str(tmp_path),
            "--release", "2.3.0",
            "--require-artifacts",
        ])
