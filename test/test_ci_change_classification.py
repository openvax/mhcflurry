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

"""Unit tests for #471 CI change classification and #359 Pages workflow."""

import importlib.util
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_classifier():
    script_path = REPO_ROOT / "scripts" / "classify_ci_changes.py"
    spec = importlib.util.spec_from_file_location("classify_ci_changes", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "path",
    [
        "docs/experiments.md",
        "docs/_static/release-2.5.13/fig.svg",
        "release-notes/2.5.13.md",
        "README.md",
        "CONTRIBUTING.md",
        "AGENTS.md",
        "Dockerfile",
        ".dockerignore",
        "docker/smoke.py",
        "mhcflurry/version.py",
        ".github/workflows/docs.yml",
        ".github/workflows/docker.yml",
    ],
)
def test_471_light_paths_allowlisted(path):
    mod = _load_classifier()
    assert mod.is_light_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "mhcflurry/common.py",
        "mhcflurry/downloads.yml",
        "mhcflurry/class1_neural_network.py",
        "test/test_packaging.py",
        "pyproject.toml",
        "requirements.txt",
        "scripts/classify_ci_changes.py",
        ".github/workflows/ci.yml",
        "downloads-generation/allele_sequences/GENERATE.sh",
        "lint.sh",
        "test.sh",
        "mhcflurry/README.md",
    ],
)
def test_471_code_test_config_paths_trigger_full(path):
    mod = _load_classifier()
    assert mod.is_light_path(path) is False


def test_471_classify_docs_and_container_pr_is_light():
    mod = _load_classifier()
    res = mod.classify_changed_files(
        [
            "docs/experiments.md",
            "docs/changelog.md",
            "release-notes/2.5.13.md",
            "mhcflurry/version.py",
            "Dockerfile",
            "docker/smoke.py",
        ],
        event_name="pull_request",
    )
    assert res["mode"] == "light"
    assert res["full_trigger_files"] == []


def test_471_classify_mixed_pr_triggers_full():
    mod = _load_classifier()
    res = mod.classify_changed_files(
        [
            "docs/changelog.md",
            "mhcflurry/version.py",
            "mhcflurry/common.py",
        ],
        event_name="pull_request",
    )
    assert res["mode"] == "full"
    assert res["full_trigger_files"] == ["mhcflurry/common.py"]


def test_471_classify_push_and_empty_fail_safe_to_full():
    mod = _load_classifier()
    push_res = mod.classify_changed_files(
        ["docs/experiments.md", "mhcflurry/version.py"],
        event_name="push",
    )
    assert push_res["mode"] == "full"

    empty_res = mod.classify_changed_files([], event_name="pull_request")
    assert empty_res["mode"] == "full"


def test_471_cli_writes_github_output_and_summary(tmp_path):
    mod = _load_classifier()
    out_file = tmp_path / "github_output.txt"
    summary_file = tmp_path / "summary.md"

    rc = mod.main([
        "--event-name", "pull_request",
        "--files", "docs/changelog.md", "mhcflurry/version.py",
        "--github-output", str(out_file),
        "--summary-file", str(summary_file),
    ])
    assert rc == 0
    out_text = out_file.read_text(encoding="utf-8")
    assert "mode=light\n" in out_text
    assert "reason=" in out_text
    summary_text = summary_file.read_text(encoding="utf-8")
    assert "`light`" in summary_text
    assert "`docs/changelog.md`" in summary_text


def test_359_docs_workflow_uses_actions_pages_not_branch_push():
    workflow_path = REPO_ROOT / ".github" / "workflows" / "docs.yml"
    raw = workflow_path.read_text(encoding="utf-8")
    assert "peaceiris/actions-gh-pages" not in raw

    parsed = yaml.safe_load(raw)
    publish = parsed["jobs"]["publish"]
    assert publish["permissions"]["pages"] == "write"
    assert publish["permissions"]["id-token"] == "write"
    assert publish["environment"]["name"] == "github-pages"

    uses = [step.get("uses", "") for step in publish["steps"]]
    assert any(u.startswith("actions/configure-pages@v6") for u in uses)
    assert any(u.startswith("actions/upload-pages-artifact@v5") for u in uses)
    assert any(u.startswith("actions/deploy-pages@v5") for u in uses)
    # Must preserve previous-versions archive from gh-pages.
    assert any(
        step.get("with", {}).get("sparse-checkout") == "previous-versions"
        for step in publish["steps"]
    )
    # Concurrency must not cancel master deployments.
    assert "pull_request" in str(parsed["concurrency"]["cancel-in-progress"])


def test_471_ci_workflow_has_classify_job_and_stable_matrix():
    workflow_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    raw = workflow_path.read_text(encoding="utf-8")
    # Guard against the GitHub Actions falsy-0 ternary footgun (`&& 0 ||`).
    non_comment = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    )
    assert "&& 0 ||" not in non_comment

    parsed = yaml.safe_load(raw)
    assert "classify" in parsed["jobs"]
    classify_checkout = next(
        step
        for step in parsed["jobs"]["classify"]["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert classify_checkout["with"]["fetch-depth"] == 0
    assert classify_checkout["with"]["filter"] == "blob:none"

    build = parsed["jobs"]["build"]
    assert build["needs"] == "classify"
    assert build["strategy"]["matrix"]["python-version"] == [
        "3.11", "3.12", "3.13", "3.14",
    ]
