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

"""Tests for the unified ``mhcflurry`` CLI.

Covers dispatch, side resolution, training_stats end-to-end on
synthetic manifests, and presence of help text. Predict-running paths
(affinity + presentation) need real models on disk and live in
integration suites, not here.
"""

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import types

import numpy
import pandas
import pytest
import yaml

from mhcflurry.cli import compare_models, main as cli_main
from mhcflurry.cli import eval_command
from mhcflurry.cli import paper_figures
from mhcflurry.cli import plot_model_comparison
from mhcflurry.cli import train_command
from mhcflurry.version import __version__


def test_top_level_parser_lists_subcommands():
    parser = cli_main.build_parser()
    help_text = parser.format_help()
    assert "train" in help_text
    assert "eval" in help_text
    assert "compare-models" in help_text
    assert "plot-model-comparison" in help_text
    assert "paper-figures" in help_text


def test_allele_specific_training_count_threshold_is_inclusive():
    from mhcflurry.cli import train_allele_specific_models_command as command

    counts = pandas.Series({"HLA-A*02:01": 50, "HLA-B*07:02": 49})
    assert command._alleles_with_minimum_measurements(counts, 50) == [
        "HLA-A*02:01"
    ]

    required = [
        "--data", "train.csv",
        "--out-models-dir", "models",
        "--hyperparameters", "hyperparameters.yaml",
    ]
    for option, value in (
            ("--min-measurements-per-allele", "0"),
            ("--held-out-fraction-reciprocal", "1"),
            ("--n-models", "0"),
            ("--save-interval", "nan")):
        with pytest.raises(SystemExit):
            command.parser.parse_args(required + [option, value])


def test_training_and_selection_commands_reject_invalid_sizes_early():
    from mhcflurry.cli import calibrate_percentile_ranks_command as calibrate
    from mhcflurry.cli import select_allele_specific_models_command as select_allele
    from mhcflurry.cli import select_pan_allele_models_command as select_pan
    from mhcflurry.cli import select_processing_models_command as select_processing
    from mhcflurry.cli import train_pan_allele_models_command as train_pan

    with pytest.raises(SystemExit):
        calibrate.parser.parse_args([
            "--models-dir", "models",
            "--num-peptides-per-length", "0",
        ])
    with pytest.raises(SystemExit):
        calibrate.run([
            "--models-dir", "models",
            "--length-range", "15", "8",
        ])

    train_base = ["--out-models-dir", "models"]
    with pytest.raises(SystemExit):
        train_pan.parser.parse_args(train_base + ["--num-folds", "0"])
    for values in (("nan", "100"), ("1.1", "100"), ("0.25", "1.5")):
        with pytest.raises(SystemExit):
            train_pan.run(train_base + [
                "--held-out-measurements-per-allele-fraction-and-max",
                *values,
            ])

    selection_commands = (
        (select_pan, "--min-models-per-fold", "--max-models-per-fold"),
        (select_processing, "--min-models-per-fold", "--max-models-per-fold"),
        (select_allele, "--combined-min-models", "--combined-max-models"),
    )
    for command, minimum_option, maximum_option in selection_commands:
        base = ["--models-dir", "models", "--out-models-dir", "selected"]
        with pytest.raises(SystemExit):
            command.parser.parse_args(base + [minimum_option, "0"])
        with pytest.raises(SystemExit):
            command.run(base + [minimum_option, "3", maximum_option, "2"])


def test_train_help_runs(capsys):
    assert cli_main.main(["train", "--help"]) == 0
    captured = capsys.readouterr().out
    assert "pan-allele-release" in captured
    assert "Deployment is opt-in" in captured


def test_train_pan_allele_release_delegates(monkeypatch, tmp_path):
    script = tmp_path / "retrain_evaluate_deploy.sh"
    script.write_text("#!/usr/bin/env bash\n")
    calls = []

    def fake_call(argv):
        calls.append(argv)
        return 17

    monkeypatch.setattr(train_command, "_workflow_script_path", lambda: script)
    monkeypatch.setattr(train_command.subprocess, "call", fake_call)

    status = cli_main.main([
        "train",
        "pan-allele-release",
        "--run-dir", "runs/2.3.0",
        "--release", "2.3.0",
    ])

    assert status == 17
    assert calls == [[
        "bash",
        str(script),
        "--run-dir", "runs/2.3.0",
        "--release", "2.3.0",
    ]]


def test_release_workflow_deploy_is_opt_in_by_default(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--skip-train",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "Skipping deploy step" in output
    assert "deploy_trained_models" not in output


def _write_minimal_deployable_run(run_dir):
    model_directories = (
            "affinity/models.combined",
            "processing/models.selected.no_flank",
            "processing/models.selected.with_flanks",
            "processing/models.selected.short_flanks",
            "presentation/models")
    for relative in model_directories:
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    info = (
        "package\tmhcflurry %s\n"
        "git commit\t%s\n" % (__version__, source_commit)
    )
    for relative in model_directories:
        (run_dir / relative / "info.txt").write_text(info)
    (run_dir / "affinity/models.combined/manifest.csv").write_text(
        "model_name\nmodel\n")
    (run_dir / "affinity/models.combined/percent_ranks.csv").write_text(
        "allele\nHLA-A*02:01\n")
    for variant in ("no_flank", "with_flanks", "short_flanks"):
        (run_dir / (
            "processing/models.selected.%s/manifest.csv" % variant
        )).write_text("model_name\nmodel\n")
    (run_dir / "presentation/models/weights.csv").write_text(
        "model_name\nmodel\n")
    (run_dir / "presentation/models/percent_ranks.csv").write_text(
        "allele\nHLA-A*02:01\n")

    holdout_dir = run_dir / "release_holdout"
    holdout_dir.mkdir()
    manifest_records = {}
    for filename, header in (
            ("affinity_pmhcs.csv", "allele,peptide\n"),
            ("affinity_samples.csv", "sample_id\n"),
            ("processing_samples.csv", "sample_id\n"),
            ("presentation_samples.csv", "sample_id\n")):
        path = holdout_dir / filename
        path.write_text(header)
        manifest_records[filename] = {
            "rows": 0,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    policy_path = holdout_dir / "policy.json"
    policy_path.write_text(json.dumps({
        "schema_version": 1,
        "holdout_files": manifest_records,
    }))
    (holdout_dir / "validation.json").write_text(json.dumps({
        "schema_version": 1,
        "policy_sha256": hashlib.sha256(
            policy_path.read_bytes()).hexdigest(),
        "holdout_files": manifest_records,
        "affinity_overlap_rows": 0,
        "processing_overlap_rows": 0,
        "presentation_overlap_rows": 0,
    }))


def test_deploy_packages_only_requested_processing_variants(tmp_path):
    run_dir = tmp_path / "release-run"
    _write_minimal_deployable_run(run_dir)
    base_command = [
        "bash",
        "scripts/release/deploy_trained_models.sh",
        "--run-dir", str(run_dir),
        "--release", "2.3.0",
        "--github-release", "2.3.0",
        "--repo", ".",
        "--allow-dirty-repo",
        "--dry-run",
    ]

    default = subprocess.run(
        base_command, capture_output=True, text=True, check=True)
    default_tar = next(
        line for line in (default.stdout + default.stderr).splitlines()
        if "models_class1_processing" in line and "tar " in line
    )
    assert "models.selected.no_flank" in default_tar
    assert "models.selected.with_flanks" in default_tar
    assert "models.selected.short_flanks" not in default_tar

    all_variants = subprocess.run(
        base_command[:-1] + [
            "--processing-variants",
            "no_flank with_flanks short_flanks",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    all_tar = next(
        line for line in (all_variants.stdout + all_variants.stderr).splitlines()
        if "models_class1_processing" in line and "tar " in line
    )
    assert "models.selected.short_flanks" in all_tar


def test_deploy_rejects_artifacts_from_a_different_commit(tmp_path):
    run_dir = tmp_path / "release-run"
    _write_minimal_deployable_run(run_dir)
    (run_dir / "affinity/models.combined/info.txt").write_text(
        "package\tmhcflurry 2.3.0rc14\n"
        "git commit\tdeadbeef\n"
    )

    result = subprocess.run(
        [
            "bash", "scripts/release/deploy_trained_models.sh",
            "--run-dir", str(run_dir),
            "--release", "2.3.0",
            "--repo", ".",
            "--allow-dirty-repo",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "artifact git commit 'deadbeef'" in result.stdout + result.stderr
    assert "tar -C" not in result.stdout + result.stderr


def test_release_workflow_forwards_processing_variants_to_deploy(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--processing-variants", "with_flanks no_flank",
            "--skip-train",
            "--skip-eval",
            "--skip-plots",
            "--deploy-mode", "dry-run",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "deploy_trained_models.sh" in output
    assert "--processing-variants with_flanks\\ no_flank" in output
    assert (
        "variants=with_flanks no_flank; eval_modes=with_flanks,no_flank"
    ) in output


def test_release_workflow_rejects_processing_mode_not_trained(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--processing-variants", "with_flanks no_flank",
            "--processing-modes", "with_flanks,no_flank,short_flanks",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--processing-modes requests 'short_flanks'" in output
    assert "pan_allele_release_full.sh" not in output


def test_release_workflow_rejects_undeployable_processing_subset_before_train(
        tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--processing-variants", "no_flank short_flanks",
            "--presentation-processing-with-flanks-kind", "short_flanks",
            "--deploy-mode", "dry-run",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2
    assert "requires the canonical with_flanks processing artifact" in output
    assert "pan_allele_release_full.sh" not in output


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        (
            ["--processing-variants", "with_flanks no_flank no_flank"],
            "Duplicate --processing-variants entry: no_flank",
        ),
        (
            ["--processing-modes", "with_flanks,with_flanks"],
            "Duplicate --processing-modes entry: with_flanks",
        ),
        (
            ["--processing-variants", "with_flanks short_flanks"],
            "must include no_flank for presentation training",
        ),
        (
            [
                "--processing-variants", "with_flanks no_flank",
                "--presentation-processing-with-flanks-kind", "no_flank",
            ],
            "must be with_flanks or short_flanks",
        ),
    ],
)
def test_release_workflow_rejects_invalid_processing_configuration_before_train(
        tmp_path, extra_args, expected_error):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--dry-run",
        ] + extra_args,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2
    assert expected_error in output
    assert "pan_allele_release_full.sh" not in output


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        (
            ["--processing-minibatch-size", "0"],
            "--processing-minibatch-size must be a positive integer",
        ),
        (
            ["--processing-held-out-samples", "-1"],
            "--processing-held-out-samples must be a positive integer",
        ),
        (
            ["--presentation-decoys-per-hit", "nan"],
            "--presentation-decoys-per-hit must be a finite positive number",
        ),
        (
            ["--presentation-feature-chunk-size", "0"],
            "--presentation-feature-chunk-size must be a positive integer",
        ),
    ],
)
def test_release_workflow_rejects_invalid_training_sizes_before_train(
        tmp_path, extra_args, expected_error):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--dry-run",
        ] + extra_args,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 2
    assert expected_error in output
    assert "pan_allele_release_full.sh" not in output


def test_release_workflow_prepare_command_is_dry_run_visible(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--skip-train",
            "--skip-eval",
            "--paper-figures-scores-dir", str(tmp_path / "paper-inputs"),
            "--paper-figures-prepare-command", "echo prepare-external-preds",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "Paper inputs:  local prepare command configured" in output
    assert "bash -lc echo\\ prepare-external-preds" in output
    assert "mhcflurry eval plot-comparison" in output


def test_release_workflow_eval_max_benchmark_files_is_forwarded(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--skip-train",
            "--skip-plots",
            "--compare-include", "affinity",
            "--eval-max-benchmark-files", "1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "mhcflurry eval compare-models" in output
    assert "--include affinity" in output
    assert "--limit-files 1" in output


def test_release_workflow_plots_include_paper_figures_by_default(tmp_path):
    run_dir = tmp_path / "release-run"
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(run_dir),
            "--release", "2.3.0",
            "--backend", "local",
            "--skip-train",
            "--skip-eval",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "mhcflurry eval plot-comparison" in output
    assert "--paper-figures-out %s" % (
        run_dir / "eval_comparison/plots/paper_figures"
    ) in output
    assert "--paper-figures-scores-dir %s" % (
        run_dir / "eval_comparison"
    ) in output


def test_release_workflow_honors_repo_env_override(tmp_path):
    env = dict(os.environ)
    env["REPO"] = str(tmp_path / "source-tree")
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "REPO=%s" % (tmp_path / "source-tree") in output


def test_release_workflow_ssh_preflight_is_dry_run_visible(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "ssh",
            "--remote", "training-host",
            "--remote-repo", "/remote/mhcflurry",
            "--remote-run-dir", "/remote/run",
            "--no-sync-remote-output",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "actual_commit=$(git -C" in output
    assert "status --porcelain --untracked-files=no" in output
    assert output.index("actual_commit=$(git -C") < output.index(
        "pan_allele_release_full.sh")
    assert "MHCFLURRY_RELEASE_GIT_COMMIT=\\$\\(git" in output


@pytest.mark.parametrize(
    ("remote_commit_matches", "remote_dirty", "expected_error"),
    [
        (False, False, "remote checkout commit deadbeef does not match"),
        (True, True, "remote checkout has tracked changes"),
    ],
)
def test_release_workflow_ssh_rejects_unverified_source(
        tmp_path, remote_commit_matches, remote_dirty, expected_error):
    source_repo = tmp_path / "source-repo"
    (source_repo / "mhcflurry").mkdir(parents=True)
    (source_repo / "mhcflurry" / "version.py").write_text(
        '__version__ = "2.3.0rc14"\n')
    subprocess.run(["git", "init", str(source_repo)], check=True)
    subprocess.run(
        ["git", "-C", str(source_repo), "add", "mhcflurry/version.py"],
        check=True,
    )
    subprocess.run(
        [
            "git", "-C", str(source_repo),
            "-c", "user.name=Test",
            "-c", "user.email=test@example.com",
            "commit", "-m", "test source",
        ],
        capture_output=True,
        check=True,
    )
    control_bin = tmp_path / "control-bin"
    remote_bin = tmp_path / "remote-bin"
    control_bin.mkdir()
    remote_bin.mkdir()
    ssh_log = tmp_path / "ssh.log"
    ssh = control_bin / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$2\" >> \"$SSH_LOG\"\n"
        "PATH=\"$REMOTE_BIN:$PATH\" sh -c \"$2\"\n"
    )
    ssh.chmod(0o755)
    remote_git = remote_bin / "git"
    remote_git.write_text(
        "#!/bin/sh\n"
        "case \"$3\" in\n"
        "  rev-parse) printf '%s\\n' \"$REMOTE_COMMIT\" ;;\n"
        "  status)\n"
        "    if [ \"$REMOTE_DIRTY\" = 1 ]; then\n"
        "      printf ' M mhcflurry/version.py\\n'\n"
        "    fi\n"
        "    ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    remote_git.chmod(0o755)
    local_commit = subprocess.check_output(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    env = dict(os.environ)
    env.update({
        "PATH": "%s:%s" % (control_bin, env["PATH"]),
        "REPO": str(source_repo),
        "SSH_LOG": str(ssh_log),
        "REMOTE_BIN": str(remote_bin),
        "REMOTE_COMMIT": local_commit if remote_commit_matches else "deadbeef",
        "REMOTE_DIRTY": "1" if remote_dirty else "0",
    })

    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "ssh",
            "--remote", "training-host",
            "--remote-repo", "/remote/mhcflurry",
            "--remote-run-dir", "/remote/run",
            "--no-sync-remote-output",
            "--skip-eval",
            "--skip-plots",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert expected_error in result.stdout + result.stderr
    commands = ssh_log.read_text()
    assert commands.count("bash -c") == 1
    assert "pan_allele_release_full.sh" not in commands


def test_release_workflow_ssh_no_sync_validates_models_remotely(tmp_path):
    source_repo = tmp_path / "source-repo"
    (source_repo / "mhcflurry").mkdir(parents=True)
    (source_repo / "mhcflurry" / "version.py").write_text(
        '__version__ = "2.3.0rc14"\n')
    subprocess.run(["git", "init", str(source_repo)], check=True)
    subprocess.run(
        ["git", "-C", str(source_repo), "add", "mhcflurry/version.py"],
        check=True,
    )
    subprocess.run(
        [
            "git", "-C", str(source_repo),
            "-c", "user.name=Test",
            "-c", "user.email=test@example.com",
            "commit", "-m", "test source",
        ],
        capture_output=True,
        check=True,
    )
    ssh_log = tmp_path / "ssh.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$2\" >> \"$SSH_LOG\"\n"
        "case \"$2\" in\n"
        "  *ssh_source_provenance*) exit 2 ;;\n"
        "  *'actual_commit=$(git -C'*)\n"
        "    printf '%s\\n' 'Verified remote source provenance' ;;\n"
        "esac\n"
    )
    ssh.chmod(0o755)
    env = dict(os.environ)
    env.update({
        "PATH": "%s:%s" % (fake_bin, env["PATH"]),
        "REPO": str(source_repo),
        "SSH_LOG": str(ssh_log),
    })

    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "ssh",
            "--remote", "training-host",
            "--remote-repo", "/remote/mhcflurry",
            "--remote-run-dir", "/remote/run",
            "--no-sync-remote-output",
            "--skip-eval",
            "--skip-plots",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    commands = ssh_log.read_text()
    assert "pan_allele_release_full.sh" in commands
    assert "validate_release_provenance.py" in commands
    assert "--run-dir' '/remote/run" in commands
    local_provenance = json.loads(
        (tmp_path / "release-run" / "release_provenance.json").read_text())
    assert local_provenance["artifacts"] == {}
    assert "step=model_provenance " not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        ([], "requires --skip-eval"),
        (["--skip-eval"], "requires --skip-plots"),
        (
            ["--skip-eval", "--skip-plots", "--deploy-mode", "dry-run"],
            "requires --deploy-mode none",
        ),
    ],
)
def test_release_workflow_no_sync_rejects_local_stages_before_training(
        tmp_path, extra_args, expected_error):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "ssh",
            "--no-sync-remote-output",
        ] + extra_args,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert expected_error in result.stdout + result.stderr
    assert "pan_allele_release_full.sh" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("cleanup_args", "expected_error"),
    [
        (
            ["--brev-on-finish", "delete"],
            "cannot be combined with --brev-on-finish delete",
        ),
        (
            ["--brev-stop-failure-action", "delete"],
            "cannot be combined with --brev-stop-failure-action delete",
        ),
    ],
)
def test_release_workflow_no_sync_preserves_only_brev_artifact_copy(
        tmp_path, cleanup_args, expected_error):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "brev-provision",
            "--brev-instance", "test-no-sync",
            "--no-sync-remote-output",
            "--skip-eval",
            "--skip-plots",
        ] + cleanup_args,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert expected_error in result.stdout + result.stderr


def test_release_workflow_forwards_processing_parallelism(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--skip-eval",
            "--skip-plots",
            "--processing-num-jobs", "4",
            "--processing-max-workers-per-gpu", "1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "PROCESSING_NUM_JOBS=4" in output
    assert "PROCESSING_MAX_WORKERS_PER_GPU=1" in output
    assert "jobs=4; workers/gpu=1" in output


def test_release_workflow_forwards_presentation_recipe_controls(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "local",
            "--skip-eval",
            "--skip-plots",
            "--processing-held-out-samples", "17",
            "--presentation-decoys-per-hit", "7",
            "--presentation-feature-chunk-size", "12345",
            "--presentation-num-jobs", "8",
            "--presentation-max-workers-per-gpu", "2",
            "--presentation-calibration-num-jobs", "3",
            "--presentation-calibration-max-workers-per-gpu", "4",
            "--presentation-calibration-prediction-batch-size", "4096",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    for expected in (
            "PROCESSING_HELD_OUT_SAMPLES=17",
            "PRESENTATION_DECOYS_PER_HIT=7",
            "PRESENTATION_FEATURE_CHUNK_SIZE=12345",
            "PRESENTATION_NUM_JOBS=8",
            "PRESENTATION_MAX_WORKERS_PER_GPU=2",
            "PRESENTATION_CALIBRATION_NUM_JOBS=3",
            "PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU=4",
            "PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE=4096"):
        assert expected in output


def test_release_workflow_sync_is_workflow_id_scoped():
    script = pathlib.Path(
        "scripts/release/retrain_evaluate_deploy.sh").read_text()
    assert 'RUNPLZ_REQUIRED_VERSION="3.15.3"' in script
    assert "require_clean_runplz_3153" in script
    assert "run_dir_matches_workflow || return 1" in script
    assert "remote_workflow_id" in script
    assert "Refusing to sync Brev output for workflow" in script
    assert "add_path .runplz/mhcflurry_release_workflow_id" in script
    assert "add_path .runplz/mhcflurry_release_workflow_exit_code" in script


def test_release_workflow_validates_selected_runplz_interpreter(tmp_path):
    fake_checkout = tmp_path / "mhcflurry-checkout"
    selected_environment = fake_checkout / ".venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(selected_environment)],
        check=True,
    )
    selected_python = selected_environment / "bin" / "python"
    site_packages = (
        selected_environment
        / "lib"
        / ("python%d.%d" % sys.version_info[:2])
        / "site-packages"
    )
    package_dir = site_packages / "runplz"
    distribution_dir = site_packages / "runplz-3.15.3.dist-info"
    package_dir.mkdir(parents=True)
    distribution_dir.mkdir()
    (fake_checkout / ".git").mkdir()
    (package_dir / "__init__.py").write_text("")
    (distribution_dir / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: runplz\nVersion: 3.15.3\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    runplz = fake_bin / "runplz"
    runplz.write_text(
        "#!%s\nimport sys\nsys.exit(23)\n" % selected_python
    )
    runplz.chmod(0o755)
    brev = fake_bin / "brev"
    brev.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = ls ]; then printf '[]\\n'; fi\n"
    )
    brev.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = "%s:%s" % (fake_bin, env["PATH"])

    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "brev-existing",
            "--brev-instance", "missing-test-instance",
            "--no-sync-remote-output",
            "--skip-eval",
            "--skip-plots",
            "--allow-dirty-repo",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 23, output
    assert "runplz provenance:" in output
    assert "executable=%s" % runplz in output
    assert "module=%s" % (package_dir / "__init__.py") in output
    assert "from PyPI or a clean checkout is required" not in output


def test_brev_postprocess_archive_includes_release_holdout(tmp_path):
    run_dir = tmp_path / "release-run"
    _write_minimal_deployable_run(run_dir)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    brev = fake_bin / "brev"
    brev.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = ls ]; then printf '[]\\n'; fi\n"
    )
    brev.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = "%s:%s" % (fake_bin, env["PATH"])

    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(run_dir),
            "--release", "2.3.0",
            "--backend", "brev-existing",
            "--brev-instance", "missing-test-instance",
            "--skip-train",
            "--skip-plots",
            "--allow-dirty-repo",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    archive = run_dir / ".brev-postprocess" / "model_artifacts.tar.bz2"
    assert archive.is_file()
    with tarfile.open(archive, "r:bz2") as tar:
        archived_paths = set(tar.getnames())
    assert {
        "release_holdout/policy.json",
        "release_holdout/validation.json",
        "release_holdout/affinity_pmhcs.csv",
        "release_holdout/affinity_samples.csv",
        "release_holdout/processing_samples.csv",
        "release_holdout/presentation_samples.csv",
    }.issubset(archived_paths)


def test_release_sync_archive_preserves_comparison_predictions(tmp_path):
    workflow = pathlib.Path(
        "scripts/release/retrain_evaluate_deploy.sh").read_text()
    marker = "cat > \"$sync_script\" <<'EOF'\n"
    archive_script = workflow.split(marker, 1)[1].split("\nEOF", 1)[0]

    out_dir = tmp_path / "runplz-latest" / "out"
    expected = [
        "eval_comparison/affinity/predictions.csv.bz2",
        "eval_comparison/processing/predictions_with_flanks.csv.bz2",
        "eval_comparison/presentation/predictions_without_flanks.csv.bz2",
    ]
    for relative in expected:
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"comparison predictions")

    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    subprocess.run(
        ["bash"],
        input=archive_script,
        text=True,
        env=env,
        capture_output=True,
        check=True,
    )

    manifest = (
        out_dir / ".runplz" / "release_sync_paths.txt"
    ).read_text().splitlines()
    assert set(expected).issubset(manifest)
    assert workflow.count(
        "add_glob eval_comparison/*/predictions*.csv.bz2"
    ) == 2


def test_release_workflow_brev_prepare_uses_remote_postprocess(tmp_path):
    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run"),
            "--release", "2.3.0",
            "--backend", "brev-provision",
            "--brev-instance", "mhcflurry-dry-run-test",
            "--paper-figures-scores-dir", str(tmp_path / "paper-inputs"),
            "--paper-figures-prepare-command", "echo prepare-external-preds",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "plotting will run in a Brev postprocess after sync" in output
    assert "Would run Brev postprocess-only eval/plot" in output
    assert "Using plots produced on the Brev instance" in output


def test_brev_postprocess_reuses_training_python_without_compile_fanout():
    workflow = pathlib.Path(
        "scripts/release/retrain_evaluate_deploy.sh").read_text()

    assert "if [ -x /opt/conda/bin/python ]; then" in workflow
    assert 'export PATH="/opt/conda/bin:$PATH"' in workflow
    assert (
        'export MHCFLURRY_TORCH_COMPILE="${MHCFLURRY_TORCH_COMPILE:-0}"'
        in workflow
    )
    assert (
        'export COMPARE_TORCH_COMPILE=%q\\n' in workflow
    )


def test_release_workflow_brev_provider_aliases(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run-default"),
            "--release", "2.3.0",
            "--backend", "brev-provision",
            "--brev-instance", "mhcflurry-dry-run-default",
            "--skip-train",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    default_output = result.stdout + result.stderr
    assert "Brev provider: auto" in default_output
    assert "Brev type:     runplz auto-select" in default_output

    cases = [
        ("auto", "Brev type:     runplz auto-select"),
        ("gcp", "Brev type:     a2-highgpu-4g:nvidia-tesla-a100:4"),
        ("denvr", "Brev type:     denvr_A100_sxm4x8"),
    ]
    for provider, expected_type in cases:
        result = subprocess.run(
            [
                "bash",
                "scripts/release/retrain_evaluate_deploy.sh",
                "--run-dir", str(tmp_path / ("release-run-" + provider)),
                "--release", "2.3.0",
                "--backend", "brev-provision",
                "--brev-instance", "mhcflurry-dry-run-" + provider,
                "--brev-provider", provider,
                "--skip-train",
                "--skip-eval",
                "--skip-plots",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout + result.stderr
        assert "Brev provider: %s" % provider in output
        assert expected_type in output


def test_release_workflow_exact_brev_type_overrides_implicit_provider(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run-exact-type"),
            "--release", "2.3.0",
            "--backend", "brev-provision",
            "--brev-instance", "mhcflurry-dry-run-exact-type",
            "--brev-instance-type", "test.gpu",
            "--skip-train",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "Brev provider: auto" in output
    assert "Brev type:     test.gpu" in output


def test_release_workflow_rejects_conflicting_explicit_brev_selection(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run-conflicting-type"),
            "--release", "2.3.0",
            "--backend", "brev-provision",
            "--brev-provider", "gcp",
            "--brev-instance-type", "test.gpu",
            "--skip-train",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "non-auto --brev-provider cannot be combined" in (
        result.stdout + result.stderr
    )


def test_release_workflow_fast_gpu_profile(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run-fast"),
            "--release", "2.3.0",
            "--backend", "brev-provision",
            "--release-profile", "fast-8xa100",
            "--brev-instance", "mhcflurry-dry-run-fast",
            "--skip-train",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "Profile:       fast-8xa100" in output
    assert "Affinity MWPG: auto" in output
    assert "Brev provider: denvr-80gb" in output
    assert "Brev type:     denvr_A100_sxm4_80Gx8" in output


def test_release_workflow_profiles_respect_explicit_overrides(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run-fast-explicit"),
            "--release", "2.3.0",
            "--backend", "brev-provision",
            "--release-profile", "fast-8xa100",
            "--brev-instance", "mhcflurry-dry-run-fast-explicit",
            "--brev-provider", "gcp",
            "--affinity-max-workers-per-gpu", "auto",
            "--skip-train",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "Profile:       fast-8xa100" in output
    assert "Affinity MWPG: auto" in output
    assert "Brev provider: gcp" in output
    assert "Brev type:     a2-highgpu-4g:nvidia-tesla-a100:4" in output


def test_release_workflow_minimal_processing_profile(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/release/retrain_evaluate_deploy.sh",
            "--run-dir", str(tmp_path / "release-run-minimal"),
            "--release", "2.3.0",
            "--backend", "local",
            "--release-profile", "minimal-processing",
            "--skip-train",
            "--skip-eval",
            "--skip-plots",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "Profile:       minimal-processing" in output
    assert (
        "Processing:    variants=with_flanks no_flank; "
        "eval_modes=with_flanks,no_flank"
    ) in output


def test_eval_help_runs(capsys):
    cli_main.main(["eval", "--help"])
    captured = capsys.readouterr().out
    assert "paper-figures score-predictions" in captured
    assert "paper-figures external-predictors" in captured
    assert "paper-figures run" in captured
    assert "Compatibility:" in captured


def test_eval_compare_models_help_runs(capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["eval", "compare-models", "--help"])
    captured = capsys.readouterr().out
    assert "usage: mhcflurry eval compare-models" in captured
    assert "--data-dir" in captured


def test_eval_plot_comparison_help_runs(capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["eval", "plot-comparison", "--help"])
    captured = capsys.readouterr().out
    assert "usage: mhcflurry eval plot-comparison" in captured
    assert "--paper-figures-scores-dir" in captured


def test_eval_paper_figures_render_help_runs(capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["eval", "paper-figures", "render", "--help"])
    captured = capsys.readouterr().out
    assert "usage: mhcflurry eval paper-figures render" in captured
    assert "--scores-dir" in captured


def test_eval_paper_figures_score_predictions_writes_cache(tmp_path):
    pytest.importorskip("sklearn")
    predictions = tmp_path / "benchmark.multiallelic.csv"
    pandas.DataFrame([
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAK",
            "hit": 1,
            "netmhcpan4.ba": 20.0,
            "mhcflurry_production": 20.0,
        },
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAL",
            "hit": 1,
            "netmhcpan4.ba": 30.0,
            "mhcflurry_production": 30.0,
        },
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAM",
            "hit": 0,
            "netmhcpan4.ba": 900.0,
            "mhcflurry_production": 900.0,
        },
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAN",
            "hit": 0,
            "netmhcpan4.ba": 1000.0,
            "mhcflurry_production": 1000.0,
        },
    ]).to_csv(predictions, index=False)
    out = tmp_path / "accuracy_scores.multiallelic.csv"

    status = eval_command.run_argv([
        "paper-figures",
        "score-predictions",
        "--kind", "multiallelic",
        "--input", str(predictions),
        "--out", str(out),
        "--external-baselines", "netmhcpan4.ba:ba",
    ])

    assert status == 0
    scores = pandas.read_csv(out)
    assert set(scores["predictor"]) == {
        "netmhcpan4.ba",
        "mhcflurry_production",
    }
    assert set(scores["length_label"]) == {"All", "9-mer"}
    assert "percent_change_auc_ba" in scores.columns


def test_eval_paper_figures_external_predictors_adds_columns(tmp_path):
    benchmark = tmp_path / "benchmark.csv"
    pandas.DataFrame([
        {
            "sample_id": "s1",
            "peptide": "PEPTIDEA",
            "hit": 1,
            "hla": "HLA-A*02:01 HLA-B*07:02",
        },
        {
            "sample_id": "s1",
            "peptide": "PEPTIDEB",
            "hit": 0,
            "hla": "HLA-A*02:01 HLA-B*07:02",
        },
    ]).to_csv(benchmark, index=False)
    fake_mhctools = tmp_path / "fake_mhctools.py"
    fake_mhctools.write_text("""#!/usr/bin/env python3
import sys
import pandas

out = sys.argv[sys.argv.index("--output-csv") + 1]
pandas.DataFrame([
    {"peptide": "PEPTIDEA", "allele": "HLA-A*02:01", "affinity": 50.0, "score": 0.9, "percentile_rank": 0.1},
    {"peptide": "PEPTIDEA", "allele": "HLA-B*07:02", "affinity": 500.0, "score": 0.1, "percentile_rank": 5.0},
    {"peptide": "PEPTIDEB", "allele": "HLA-A*02:01", "affinity": 1000.0, "score": 0.2, "percentile_rank": 10.0},
    {"peptide": "PEPTIDEB", "allele": "HLA-B*07:02", "affinity": 30.0, "score": 0.8, "percentile_rank": 0.2},
]).to_csv(out, index=False)
""")
    fake_mhctools.chmod(0o755)
    out = tmp_path / "benchmark.with_external.csv"

    status = eval_command.run_argv([
        "paper-figures",
        "external-predictors",
        "--input", str(benchmark),
        "--out", str(out),
        "--mhctools-command", str(fake_mhctools),
        "--predictor", "fake:netmhcpan4.2.ba:affinity",
    ])

    assert status == 0
    result = pandas.read_csv(out)
    assert result["netmhcpan4.2.ba"].tolist() == [50.0, 30.0]
    assert result["netmhcpan4.2.ba_best_allele"].tolist() == [
        "HLA-A*02:01", "HLA-B*07:02"]


@pytest.mark.parametrize("invalid", ["HLA-A2", "NONSENSE"])
def test_external_predictors_reject_unnormalized_alleles(invalid):
    with pytest.raises(ValueError, match="Invalid, ambiguous, or unsupported"):
        eval_command._split_benchmark_alleles(
            "HLA-A*02:01 %s" % invalid)


@pytest.mark.parametrize("invalid", ["", "   "])
def test_external_predictors_reject_empty_genotypes(invalid):
    with pytest.raises(ValueError, match="contains no class-I alleles"):
        eval_command._split_benchmark_alleles(invalid)


@pytest.mark.parametrize(
    "predictor_specs",
    [
        ["one:score_a:score", "two:score_a:score"],
        ["one:score_a:score", "two:score_a_best_allele:score"],
        ["one:peptide:score"],
        ["one:hit:score"],
    ],
)
def test_external_predictors_rejects_output_column_collisions(
        tmp_path, predictor_specs):
    benchmark = tmp_path / "benchmark.csv"
    pandas.DataFrame([{
        "peptide": "SIINFEKL",
        "hla": "HLA-A*02:01",
        "hit": 1,
    }]).to_csv(benchmark, index=False)
    argv = [
        "--input", str(benchmark),
        "--out", str(tmp_path / "out.csv"),
    ]
    for spec in predictor_specs:
        argv.extend(["--predictor", spec])
    args = eval_command._make_external_predictors_parser(
        "external-predictors").parse_args(argv)

    with pytest.raises(ValueError, match="output column"):
        eval_command._run_external_predictors(args)


def test_external_predictor_raw_filenames_do_not_use_output_column(
        tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmark.csv"
    pandas.DataFrame([{
        "peptide": "SIINFEKL",
        "hla": "HLA-A*02:01",
        "hit": 1,
    }]).to_csv(benchmark, index=False)
    raw_dir = tmp_path / "raw"
    captured = []

    def fake_runner(_command, _predictor, _alleles, peptides, out_csv):
        captured.append(pathlib.Path(out_csv))
        pandas.DataFrame([{
            "peptide": peptides[0],
            "allele": "HLA-A*02:01",
            "score": 0.5,
        }]).to_csv(out_csv, index=False)

    monkeypatch.setattr(eval_command, "_run_mhctools_for_group", fake_runner)
    args = eval_command._make_external_predictors_parser(
        "external-predictors").parse_args([
            "--input", str(benchmark),
            "--out", str(tmp_path / "out.csv"),
            "--keep-raw-dir", str(raw_dir),
            "--predictor", "fake:../../unsafe-name:score",
        ])

    assert eval_command._run_external_predictors(args) == 0
    assert captured == [raw_dir / "predictor.0000.group.0000.csv"]
    assert not (tmp_path / "unsafe-name.0000.csv").exists()


def test_eval_paper_figures_run_dispatches_pipeline(tmp_path, monkeypatch):
    calls = []

    def fake_compare(args):
        calls.append(("compare", args.a, args.b, args.out, args.include))
        return 0

    def fake_paper(args):
        calls.append((
            "paper",
            args.comparison_dir,
            args.out,
            args.formats,
            args.scores_dir,
            args.external_baselines,
        ))
        return 0

    def fake_plot(args):
        calls.append((
            "plot",
            args.input,
            args.summary_pdf,
            args.paper_figures_out,
            args.include_paper_figures_in_summary_pdf,
        ))
        return 0

    monkeypatch.setattr(compare_models, "run", fake_compare)
    monkeypatch.setattr(paper_figures, "run", fake_paper)
    monkeypatch.setattr(plot_model_comparison, "run", fake_plot)

    out = tmp_path / "eval"
    status = eval_command.run_argv([
        "paper-figures",
        "run",
        "--a", "new-run",
        "--b", "public:2.0.0",
        "--out", str(out),
        "--include", "affinity",
        "--scores-dir", str(tmp_path / "scores"),
        "--external-baselines", "netmhcpan4.ba:ba",
        "--formats", "svg,pdf",
    ])
    assert status == 0
    assert calls == [
        ("compare", "new-run", "public:2.0.0", str(out), "affinity"),
        (
            "paper",
            str(out),
            str(out / "plots" / "paper_figures"),
            "svg,pdf",
            str(tmp_path / "scores"),
            "netmhcpan4.ba:ba",
        ),
        (
            "plot",
            str(out),
            str(out / "plots" / "model_comparison_figures.pdf"),
            str(out / "plots" / "paper_figures"),
            True,
        ),
    ]


def test_eval_paper_figures_run_preserves_rendered_paper_suite(
        tmp_path, monkeypatch):
    out = tmp_path / "eval"
    combined = out / "plots" / "paper_figures" / "paper_figures.pdf"
    summary = out / "plots" / "model_comparison_figures.pdf"

    def fake_compare(args):
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        return 0

    def fake_paper(args):
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        combined.write_bytes(b"rendered paper figures")
        return 0

    def fake_summary(
            plot_dir, out_path, include_paper_figures=False,
            paper_figures_dir=None):
        assert include_paper_figures
        assert pathlib.Path(paper_figures_dir) == combined.parent
        assert combined.read_bytes() == b"rendered paper figures"
        pathlib.Path(out_path).write_bytes(b"summary")

    monkeypatch.setattr(compare_models, "run", fake_compare)
    monkeypatch.setattr(paper_figures, "run", fake_paper)
    monkeypatch.setattr(plot_model_comparison, "_apply_paper_style", lambda: None)
    monkeypatch.setattr(
        plot_model_comparison, "_plot_release_summary", lambda *args: None)
    monkeypatch.setattr(
        plot_model_comparison, "_write_summary_pdf", fake_summary)

    status = eval_command.run_argv([
        "paper-figures", "run",
        "--a", "new-run",
        "--out", str(out),
    ])

    assert status == 0
    assert combined.read_bytes() == b"rendered paper figures"
    assert summary.read_bytes() == b"summary"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--paper-figures-out", "{out}/plots"],
        [
            "--summary-pdf",
            "{out}/plots/paper_figures/paper_figures.pdf",
        ],
    ],
)
def test_eval_paper_figures_run_rejects_output_collisions_before_work(
        tmp_path, monkeypatch, extra_args):
    calls = []
    out = tmp_path / "eval"
    rendered = out / "plots" / "paper_figures.pdf"
    rendered.parent.mkdir(parents=True)
    rendered.write_bytes(b"existing output")

    monkeypatch.setattr(
        compare_models, "run", lambda args: calls.append("compare"))
    monkeypatch.setattr(
        paper_figures, "run", lambda args: calls.append("paper"))
    monkeypatch.setattr(
        plot_model_comparison, "run", lambda args: calls.append("plot"))
    argv = [
        "paper-figures", "run",
        "--a", "new-run",
        "--out", str(out),
    ] + [value.format(out=out) for value in extra_args]

    with pytest.raises(SystemExit, match="command-owned|dedicated directory"):
        eval_command.run_argv(argv)

    assert calls == []
    assert rendered.read_bytes() == b"existing output"


@pytest.mark.parametrize(
    ("option", "relative"),
    [
        ("--scores-dir", "plots/saved-scores"),
        ("--multiallelic-predictions", "affinity/saved-predictions.csv"),
        ("--monoallelic-predictions", "worker_logs/saved-predictions.csv"),
    ],
)
def test_eval_paper_figures_run_preserves_inputs_during_compare_reset(
        tmp_path, monkeypatch, option, relative):
    calls = []
    out = tmp_path / "eval"
    source = out / relative
    if source.suffix:
        source.parent.mkdir(parents=True)
        source.write_text("saved input")
    else:
        source.mkdir(parents=True)
        (source / "saved.csv").write_text("saved input")

    monkeypatch.setattr(
        compare_models, "run", lambda args: calls.append("compare"))
    monkeypatch.setattr(
        paper_figures, "run", lambda args: calls.append("paper"))
    monkeypatch.setattr(
        plot_model_comparison, "run", lambda args: calls.append("plot"))

    with pytest.raises(SystemExit, match="would be deleted"):
        eval_command.run_argv([
            "paper-figures", "run",
            "--a", "new-run",
            "--out", str(out),
            option, str(source),
        ])

    assert calls == []
    assert source.exists()


def test_compare_models_help_runs(capsys):
    """The compare-models help text exposes the documented flags.

    Goes through ``main()`` rather than ``build_parser`` because subparsers
    are name-only (lazy import); per-subcommand args are only built when
    the legacy module is actually invoked.
    """
    with pytest.raises(SystemExit):
        cli_main.main(["compare-models", "--help"])
    captured = capsys.readouterr().out
    for flag in ["--a", "--b", "--include", "--out", "--data-dir",
                 "--num-jobs", "--gpus", "--max-workers-per-gpu",
                 "--processing-modes", "--presentation-modes",
                 "--presentation-num-jobs",
                 "--presentation-max-workers-per-gpu",
                 "--presentation-torch-compile"]:
        assert flag in captured, "missing flag in help: %s" % flag


def test_compare_models_presentation_parallelism_overrides():
    parser = compare_models.make_parser()
    args = parser.parse_args([
        "--a", "public",
        "--out", "unused",
        "--num-jobs", "4",
        "--max-workers-per-gpu", "4",
        "--presentation-num-jobs", "1",
        "--presentation-max-workers-per-gpu", "1",
        "--presentation-max-tasks-per-worker", "1",
        "--presentation-torch-compile", "0",
    ])
    args._local_parallelism_args_resolved = True
    args.workload_plan = object()

    affinity_args = compare_models._parallelism_args_for_component(
        args, "affinity")
    processing_args = compare_models._parallelism_args_for_component(
        args, "processing")
    presentation_args = compare_models._parallelism_args_for_component(
        args, "presentation")

    assert affinity_args is not args
    assert affinity_args.num_jobs == 4
    assert not affinity_args._local_parallelism_args_resolved
    assert not hasattr(affinity_args, "workload_plan")
    assert processing_args is not args
    assert processing_args.num_jobs == 4
    assert not processing_args._local_parallelism_args_resolved
    assert not hasattr(processing_args, "workload_plan")
    assert presentation_args is not args
    assert presentation_args.num_jobs == 1
    assert presentation_args.max_workers_per_gpu == 1
    assert presentation_args.max_tasks_per_worker == 1
    assert presentation_args.torch_compile == "0"
    assert not presentation_args._local_parallelism_args_resolved
    assert not hasattr(presentation_args, "workload_plan")
    assert args.num_jobs == 4
    assert args.max_workers_per_gpu == 4
    assert args._local_parallelism_args_resolved


def test_plot_help_runs(capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["plot-model-comparison", "--help"])
    captured = capsys.readouterr().out
    assert "--input" in captured
    assert "--a-label" in captured
    assert "--paper-figures-scores-dir" in captured
    assert "--paper-figures-artifacts-dir" in captured
    assert "--paper-figures-multiallelic-predictions" in captured
    assert "--paper-figures-candidate-predictor" in captured
    assert "--include-paper-figures-in-summary-pdf" in captured


def test_plot_model_comparison_dispatches_paper_figures(tmp_path, monkeypatch):
    captured = {}

    def fake_run(args):
        captured["scores_dir"] = args.scores_dir
        captured["comparison_dir"] = args.comparison_dir
        captured["out"] = args.out
        captured["formats"] = args.formats
        captured["candidate_predictor"] = args.candidate_predictor
        captured["external_baselines"] = args.external_baselines
        captured["multiallelic_predictions"] = args.multiallelic_predictions
        return 0

    monkeypatch.setattr(paper_figures, "run", fake_run)
    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(tmp_path / "comparison"),
        "--paper-figures-scores-dir", str(tmp_path / "scores"),
        "--paper-figures-out", str(tmp_path / "paper"),
        "--paper-figures-formats", "svg,pdf",
        "--paper-figures-candidate-predictor", "candidate",
        "--paper-figures-external-baselines", "baseline:ba",
        "--paper-figures-multiallelic-predictions",
        str(tmp_path / "predictions.csv"),
    ])
    assert plot_model_comparison.run(args) == 0
    assert captured == {
        "scores_dir": str(tmp_path / "scores"),
        "comparison_dir": str(tmp_path / "comparison"),
        "out": str(tmp_path / "paper"),
        "formats": "svg,pdf",
        "candidate_predictor": "candidate",
        "external_baselines": "baseline:ba",
        "multiallelic_predictions": str(tmp_path / "predictions.csv"),
    }


def test_paper_figures_help_runs(capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["paper-figures", "--help"])
    captured = capsys.readouterr().out
    assert "--scores-dir" in captured
    assert "--artifacts-dir" in captured
    assert "--comparison-dir" in captured
    assert "--multiallelic-predictions" in captured
    assert "--formats" in captured
    assert "--candidate-predictor" in captured
    assert "--external-baselines" in captured


def test_unknown_subcommand_errors():
    parser = cli_main.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["does-not-exist"])


def test_all_legacy_commands_registered():
    """All 12 legacy mhcflurry-* commands are reachable as subcommands."""
    expected = {
        "predict", "predict-scan", "downloads",
        "calibrate-percentile-ranks",
        "class1-train-allele-specific-models",
        "class1-select-allele-specific-models",
        "class1-train-pan-allele-models",
        "class1-select-pan-allele-models",
        "class1-train-processing-models",
        "class1-select-processing-models",
        "class1-train-presentation-models",
        "pseudosequences",
    }
    assert expected.issubset(set(cli_main._SUBCOMMANDS))


def _load_script_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_affinity_hyperparameter_generator_is_importable():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "scripts/training/release_exact/generate_hyperparameters.py",
    )
    module = _load_script_module(path, "affinity_hyperparameters_under_test")
    grid = module.build_grid(minibatch_size=2048)
    assert len(grid) == 35
    assert {item["minibatch_size"] for item in grid} == {2048}


def test_processing_hyperparameter_generator_is_importable():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "scripts/training/release_exact/generate_hyperparameters.base.py",
    )
    module = _load_script_module(path, "processing_hyperparameters_under_test")
    grid = module.build_grid(minibatch_size=2048)
    assert len(grid) == 128
    assert {item["minibatch_size"] for item in grid} == {2048}


def test_training_hyperparameter_cli_generates_processing_variant(tmp_path, capsys):
    cli_main.main([
        "class1-generate-training-hyperparameters",
        "processing-base",
        "--minibatch-size",
        "2048",
    ])
    base_text = capsys.readouterr().out
    base = yaml.safe_load(base_text)
    assert len(base) == 128
    assert {item["minibatch_size"] for item in base} == {2048}

    base_path = tmp_path / "hyperparameters.base.yaml"
    base_path.write_text(base_text)
    cli_main.main([
        "class1-generate-training-hyperparameters",
        "processing-variant",
        str(base_path),
        "short_flanks",
    ])
    variant = yaml.safe_load(capsys.readouterr().out)
    assert len(variant) == 128
    assert {item["n_flank_length"] for item in variant} == {5}
    assert {item["c_flank_length"] for item in variant} == {5}


def test_training_hyperparameter_helpers_reject_invalid_configuration():
    from mhcflurry.cli import generate_training_hyperparameters as generator

    with pytest.raises(ValueError, match="Unknown processing variant"):
        generator.transform_processing_hyperparameters(
            "typo", {"n_flank_length": 15, "c_flank_length": 15})
    with pytest.raises(SystemExit):
        generator.make_parser().parse_args([
            "affinity", "--minibatch-size", "0",
        ])


def test_reassign_mass_spec_training_data_cli(tmp_path):
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "out.csv"
    pandas.DataFrame({
        "measurement_kind": ["mass_spec", "binding"],
        "measurement_inequality": ["=", "="],
        "measurement_value": [123.0, 456.0],
    }).to_csv(input_path, index=False)

    cli_main.main([
        "class1-reassign-mass-spec-training-data",
        str(input_path),
        "--set-measurement-value",
        "100",
        "--out-csv",
        str(output_path),
    ])
    result = pandas.read_csv(output_path)
    assert result.measurement_value.tolist() == [100.0, 456.0]


def test_remote_launcher_preserves_shared_minibatch_override(monkeypatch):
    """Family-specific minibatch env vars should only be set when provided."""
    fake_runplz = types.ModuleType("runplz")
    fake_config = types.ModuleType("runplz.config")

    pip_packages = []

    class FakeImage:
        @classmethod
        def from_registry(cls, *_args, **_kwargs):
            return cls()

        def apt_install(self, *_args, **_kwargs):
            return self

        def pip_install(self, *args, **_kwargs):
            pip_packages.extend(args)
            return self

        def pip_install_local_dir(self, *_args, **_kwargs):
            return self

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, *_args, **_kwargs):
            return lambda fn: fn

        def local_entrypoint(self, *_args, **_kwargs):
            return lambda fn: fn

    fake_runplz.App = FakeApp
    fake_runplz.Image = FakeImage
    fake_config.BrevConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "runplz", fake_runplz)
    monkeypatch.setitem(sys.modules, "runplz.config", fake_config)

    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "scripts/training/launch_pan_allele_training_remote.py",
    )
    module = _load_script_module(path, "remote_launcher_under_test")
    assert "runplz==3.15.3" in pip_packages
    env = module.remote_training_env({"TRAINING_MINIBATCH_SIZE": "2048"})
    assert env["TRAINING_MINIBATCH_SIZE"] == "2048"
    assert env["COMPARE_BASELINE"] == "public:2.0.0"
    assert env["COMPARE_BASELINE_LABEL"] == "MHCflurry 2.0"
    assert env["COMPARE_BACKEND"] == "auto"
    assert env["EVAL_MAX_BENCHMARK_FILES"] == ""
    assert env["COMPARE_GPUS"] == "auto"
    assert env["COMPARE_TORCH_COMPILE"] == "auto"
    assert env["COMPARE_MATMUL_PRECISION"] == "high"
    assert env["MHCFLURRY_RELEASE_WORKFLOW_ID"] == ""
    assert env["MHCFLURRY_RELEASE_GIT_COMMIT"] == ""
    assert env["MHCFLURRY_RELEASE_VERSION"] == ""
    assert env["MHCFLURRY_GPU_TELEMETRY"] == "1"
    assert env["MHCFLURRY_GPU_TELEMETRY_SECONDS"] == "30"
    assert env["NUM_JOBS"] == "auto"
    assert env["MKL_THREADING_LAYER"] == "GNU"
    assert env["COMPARE_PRESENTATION_NUM_JOBS"] == "auto"
    assert env["COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU"] == "auto"
    assert env["COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER"] == "1"
    assert env["COMPARE_PRESENTATION_TORCH_COMPILE"] == "0"
    assert "AFFINITY_MINIBATCH_SIZE" not in env
    assert "AFFINITY_MAX_WORKERS_PER_GPU" not in env
    assert "PROCESSING_MINIBATCH_SIZE" not in env
    assert env["PROCESSING_NUM_JOBS"] == "auto"
    assert env["PROCESSING_MAX_WORKERS_PER_GPU"] == "auto"
    assert env["PROCESSING_HELD_OUT_SAMPLES"] == "50"
    assert env["PRESENTATION_DECOYS_PER_HIT"] == "99"
    assert env["PRESENTATION_FEATURE_CHUNK_SIZE"] == "250000"
    assert env["PRESENTATION_NUM_JOBS"] == "auto"
    assert env["PRESENTATION_MAX_WORKERS_PER_GPU"] == "auto"
    assert env["PRESENTATION_CALIBRATION_NUM_JOBS"] == "auto"
    assert env["PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU"] == "auto"
    assert env["PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE"] == "auto"

    env = module.remote_training_env({
        "TRAINING_MINIBATCH_SIZE": "2048",
        "AFFINITY_MINIBATCH_SIZE": "512",
        "AFFINITY_MAX_WORKERS_PER_GPU": "3",
        "PROCESSING_NUM_JOBS": "4",
        "PROCESSING_MAX_WORKERS_PER_GPU": "1",
        "PROCESSING_HELD_OUT_SAMPLES": "17",
        "PRESENTATION_DECOYS_PER_HIT": "7",
        "PRESENTATION_FEATURE_CHUNK_SIZE": "12345",
        "PRESENTATION_NUM_JOBS": "8",
        "PRESENTATION_MAX_WORKERS_PER_GPU": "2",
        "PRESENTATION_CALIBRATION_NUM_JOBS": "3",
        "PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU": "4",
        "PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE": "4096",
        "COMPARE_BASELINE": "public:2.2.0",
        "COMPARE_BASELINE_LABEL": "MHCflurry 2.2",
        "COMPARE_BACKEND": "cpu",
        "EVAL_MAX_BENCHMARK_FILES": "1",
        "COMPARE_GPUS": "1",
        "COMPARE_TORCH_COMPILE": "off",
        "COMPARE_MATMUL_PRECISION": "medium",
        "MHCFLURRY_RELEASE_WORKFLOW_ID": "run-123",
        "MHCFLURRY_RELEASE_GIT_COMMIT": "abc123",
        "MHCFLURRY_RELEASE_VERSION": "2.3.0",
        "MHCFLURRY_GPU_TELEMETRY": "0",
        "MHCFLURRY_GPU_TELEMETRY_SECONDS": "5",
        "NUM_JOBS": "6",
        "MKL_THREADING_LAYER": "TBB",
    })
    assert env["AFFINITY_MINIBATCH_SIZE"] == "512"
    assert env["AFFINITY_MAX_WORKERS_PER_GPU"] == "3"
    assert env["PROCESSING_NUM_JOBS"] == "4"
    assert env["PROCESSING_MAX_WORKERS_PER_GPU"] == "1"
    assert env["PROCESSING_HELD_OUT_SAMPLES"] == "17"
    assert env["PRESENTATION_DECOYS_PER_HIT"] == "7"
    assert env["PRESENTATION_FEATURE_CHUNK_SIZE"] == "12345"
    assert env["PRESENTATION_NUM_JOBS"] == "8"
    assert env["PRESENTATION_MAX_WORKERS_PER_GPU"] == "2"
    assert env["PRESENTATION_CALIBRATION_NUM_JOBS"] == "3"
    assert env["PRESENTATION_CALIBRATION_MAX_WORKERS_PER_GPU"] == "4"
    assert env["PRESENTATION_CALIBRATION_PREDICTION_BATCH_SIZE"] == "4096"
    assert env["COMPARE_BASELINE"] == "public:2.2.0"
    assert env["COMPARE_BASELINE_LABEL"] == "MHCflurry 2.2"
    assert env["COMPARE_BACKEND"] == "cpu"
    assert env["EVAL_MAX_BENCHMARK_FILES"] == "1"
    assert env["COMPARE_GPUS"] == "1"
    assert env["COMPARE_TORCH_COMPILE"] == "off"
    assert env["COMPARE_MATMUL_PRECISION"] == "medium"
    assert env["MHCFLURRY_RELEASE_WORKFLOW_ID"] == "run-123"
    assert env["MHCFLURRY_RELEASE_GIT_COMMIT"] == "abc123"
    assert env["MHCFLURRY_RELEASE_VERSION"] == "2.3.0"
    assert env["MHCFLURRY_GPU_TELEMETRY"] == "0"
    assert env["MHCFLURRY_GPU_TELEMETRY_SECONDS"] == "5"
    assert env["NUM_JOBS"] == "6"
    assert env["MKL_THREADING_LAYER"] == "TBB"

    brev_config = module.brev_config_from_env({
        "RUNPLZ_BREV_AUTO_CREATE": "1",
        "RUNPLZ_BREV_INSTANCE_TYPE": "test.gpu",
        "RUNPLZ_BREV_ON_FINISH": "stop",
        "RUNPLZ_BREV_MAX_RUNTIME_SECONDS": "3600",
        "RUNPLZ_BREV_INSTANCE_TYPE_FALLBACK_COUNT": "5",
        "RUNPLZ_BREV_EXCLUDE_PROVIDERS": "oci,broken",
    })
    assert brev_config["auto_create_instances"] is True
    assert brev_config["instance_type"] == "test.gpu"
    assert brev_config["on_finish"] == "stop"
    assert brev_config["max_runtime_seconds"] == 3600
    assert brev_config["instance_type_fallback_count"] == 5
    assert brev_config["exclude_providers"] == ("oci", "broken")

    assert module.compare_torch_compile_value({
        "MHCFLURRY_TORCH_COMPILE": "yes",
    }) == "1"
    assert module.compare_torch_compile_value({
        "COMPARE_TORCH_COMPILE": "off",
    }) == "0"
    assert module.compare_torch_compile_value({}) == "auto"
    assert module.compare_matmul_precision_value({
        "COMPARE_MATMUL_PRECISION": "HIGH",
    }) == "high"
    with pytest.raises(ValueError):
        module.compare_torch_compile_value({"COMPARE_TORCH_COMPILE": "maybe"})
    with pytest.raises(ValueError):
        module.compare_matmul_precision_value({
            "COMPARE_MATMUL_PRECISION": "fast",
        })


def test_main_help_does_not_import_predict_command():
    """``mhcflurry --help`` must not pay the cost of importing every
    legacy command module. Lazy-import is the whole reason build_parser
    only registers subcommand names."""
    import subprocess
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, "-c",
         "import sys; from mhcflurry.cli.main import build_parser; "
         "build_parser(); "
         "print('\\n'.join(name for name in ("
         "'mhcflurry.predict_command', "
         "'mhcflurry.cli.predict_command') if name in sys.modules))"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "", (
        "predict_command was imported by build_parser(); should be lazy: %s"
        % result.stdout
    )


def test_legacy_command_module_shims_reexport_cli_modules():
    command_modules = [
        "calibrate_percentile_ranks_command",
        "downloads_command",
        "predict_command",
        "predict_scan_command",
        "select_allele_specific_models_command",
        "select_pan_allele_models_command",
        "select_processing_models_command",
        "train_allele_specific_models_command",
        "train_pan_allele_models_command",
        "train_processing_models_command",
        "train_presentation_models_command",
    ]
    for module_name in command_modules:
        legacy = importlib.import_module("mhcflurry.%s" % module_name)
        canonical = importlib.import_module("mhcflurry.cli.%s" % module_name)
        assert legacy.run is canonical.run
        assert legacy.parser is canonical.parser

    from mhcflurry import predict_command
    from mhcflurry.cli import predict_command as cli_predict_command

    assert (
        predict_command._predict_dataframe_chunk
        is cli_predict_command._predict_dataframe_chunk
    )


def test_main_dispatches_pseudosequences_list(capsys):
    """End-to-end: ``mhcflurry pseudosequences list`` runs the legacy
    module's main(argv) and prints the registry rows."""
    cli_main.main(["pseudosequences", "list"])
    out = capsys.readouterr().out
    assert "netmhcpan" in out
    assert "pseudosequences.mhcflurry.39aa.csv" in out


def test_main_unknown_subcommand_exits():
    with pytest.raises(SystemExit):
        cli_main.main(["does-not-exist"])


def test_version_flag(capsys):
    """``mhcflurry --version`` and ``-V`` print the package version + exit 0."""
    from mhcflurry.version import __version__
    for flag in ("--version", "-V"):
        assert cli_main.main([flag]) == 0
        out = capsys.readouterr().out
        assert out.strip() == "mhcflurry %s" % __version__


def test_bare_invocation_shows_grouped_help(capsys):
    """Bare ``mhcflurry`` prints the grouped help screen to stdout + exits 0
    (treats no-args as 'user asked for help')."""
    assert cli_main.main([]) == 0
    out = capsys.readouterr().out
    assert "MHCflurry " in out  # version banner
    for group in (
            "Prediction:", "Calibration:", "Class I training:",
            "Class I selection:", "Evaluation and figures",
    ):
        assert group in out, "missing group header: %s" % group
    # The example block should also surface.
    assert "Examples:" in out
    assert "mhcflurry eval compare-models" in out


def test_subsubcommand_help_shows_full_prog(capsys):
    """``mhcflurry pseudosequences filename --help`` must show the full
    ``mhcflurry pseudosequences filename`` prog, not the inherited
    ``sys.argv[0]`` from the parent. Regression for sub-subparsers
    inheriting prog at parser-build time."""
    with pytest.raises(SystemExit):
        cli_main.main(["pseudosequences", "filename", "--help"])
    captured = capsys.readouterr().out
    assert "usage: mhcflurry pseudosequences filename" in captured, captured


def test_artifact_subcommand_preserves_recorded_argv(monkeypatch, tmp_path):
    """Artifact commands dispatched through ``mhcflurry`` must record a
    split executable + subcommand in GENERATE.sh, not a single argv[0]
    containing a space."""
    from mhcflurry.common import write_generate_sh

    out_dir = tmp_path / "models"
    module = types.ModuleType("test_cli_fake_artifact_command")

    def run(argv):
        assert argv == ["--out-dir", str(out_dir)]
        os.makedirs(str(out_dir))
        write_generate_sh(str(out_dir), mhcflurry_version="9.9.9-test")
        return 0

    module.run = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        cli_main._SUBCOMMANDS,
        "fake-artifact",
        (module.__name__, "run", "Fake artifact-producing command."),
    )
    monkeypatch.setattr(
        sys, "argv", ["mhcflurry", "fake-artifact", "--out-dir", str(out_dir)])

    assert cli_main.main(sys.argv[1:]) == 0

    contents = (out_dir / "GENERATE.sh").read_text()
    assert "'mhcflurry fake-artifact'" not in contents
    assert "mhcflurry \\\n    fake-artifact" in contents


def test_rewrite_parser_prog_restores_on_exit(capsys):
    """Two consecutive dispatches must each see their own prog — i.e.
    the saved/restore cycle in _rewrite_parser_prog actually undoes
    the rewrite. Catches a regression where the second call would
    leak the first call's prog."""
    with pytest.raises(SystemExit):
        cli_main.main(["pseudosequences", "filename", "--help"])
    first = capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli_main.main(["pseudosequences", "filename", "--help"])
    second = capsys.readouterr().out
    assert "usage: mhcflurry pseudosequences filename" in first
    assert "usage: mhcflurry pseudosequences filename" in second


# ---------------------------------------------------------------------------
# Side resolution
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    defaults = dict(
        a=None, b=None, a_label=None, b_label=None, out=None,
        include="auto", data_dir=None, limit_files=None,
        affinity_source="mixmhcpred",
        processing_modes="with_flanks,no_flank,short_flanks",
        presentation_modes="with_flanks,without_flanks",
    )
    for letter in ("a", "b"):
        for role in ("affinity", "processing", "presentation", "training"):
            defaults["%s_%s_dir" % (letter, role)] = None
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_side_label_defaults_to_basename(tmp_path):
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    side = compare_models._resolve_side(
        "a", str(run_dir), label=None, args=_make_args())
    assert side["label"] == "my_run"
    assert side["spec"] == str(run_dir)


def test_resolve_side_public_label():
    side = compare_models._resolve_side(
        "b", "public", label=None, args=_make_args())
    assert side["label"] == "public"


def test_resolve_side_public_pinned_release_label():
    side = compare_models._resolve_side(
        "b", "public:4-pre-2.2.0", label=None, args=_make_args())
    assert side["label"] == "public:4-pre-2.2.0"


def test_resolve_side_public_pin_does_not_leak(monkeypatch):
    from mhcflurry import downloads

    release_env = "MHCFLURRY_DOWNLOADS_CURRENT_RELEASE"
    monkeypatch.delenv(release_env, raising=False)

    def fake_configure():
        pass

    def fake_get_path(download_name, sub):
        release = os.environ.get(release_env, "current")
        return "/downloads/%s/%s/%s" % (release, download_name, sub)

    monkeypatch.setattr(downloads, "configure", fake_configure)
    monkeypatch.setattr(downloads, "get_path", fake_get_path)

    side_a = compare_models._resolve_side(
        "a", "public:pinned-release", label=None, args=_make_args())
    side_b = compare_models._resolve_side(
        "b", "public", label=None, args=_make_args())

    assert side_a["paths"]["affinity"].startswith(
        "/downloads/pinned-release/")
    assert side_b["paths"]["affinity"].startswith("/downloads/current/")
    assert release_env not in os.environ


def test_resolve_side_explicit_label_overrides_default():
    side = compare_models._resolve_side(
        "a", "public", label="baseline", args=_make_args())
    assert side["label"] == "baseline"


def test_resolve_side_publicy_path_is_not_public_sentinel(tmp_path):
    """A user-named directory like ``public_data/`` must not be mistaken
    for the public-install sentinel."""
    run_dir = tmp_path / "public_data"
    run_dir.mkdir()
    side = compare_models._resolve_side(
        "a", str(run_dir), label=None, args=_make_args())
    # Label derives from basename, not the literal "public" sentinel.
    assert side["label"] == "public_data"
    # And no role paths were resolved through the public-download lookup.
    for role, path in side["paths"].items():
        if path is not None:
            assert path.startswith(str(tmp_path)), (role, path)


def test_resolve_side_override_paths_win(tmp_path):
    override = tmp_path / "overridden_affinity"
    override.mkdir()
    args = _make_args(a_affinity_dir=str(override))
    side = compare_models._resolve_side("a", "public", None, args)
    assert side["paths"]["affinity"] == str(override)


def test_probe_run_dir_finds_training_via_manifest(tmp_path):
    """A run dir with manifest.csv inside models.unselected.combined is
    picked up for the training role even when nested under affinity/."""
    target = tmp_path / "run" / "affinity" / "models.unselected.combined"
    target.mkdir(parents=True)
    (target / "manifest.csv").write_text(
        "model_name,config_json\nmodel_a,\"{}\"\n")
    resolved = compare_models._probe_run_dir(
        str(tmp_path / "run"), "training")
    assert resolved == str(target)


def test_probe_run_dir_finds_affinity_via_allele_sequences(tmp_path):
    target = tmp_path / "run" / "affinity" / "models.combined"
    target.mkdir(parents=True)
    # The presence of allele_sequences.csv is one of the affinity probes.
    (target / "allele_sequences.csv").write_text("allele,sequence\n")
    resolved = compare_models._probe_run_dir(
        str(tmp_path / "run"), "affinity")
    assert resolved == str(target)


def test_probe_run_dir_finds_presentation_via_weights_csv(tmp_path):
    target = tmp_path / "run" / "presentation" / "models"
    target.mkdir(parents=True)
    # Class1PresentationPredictor.save() writes weights.csv at the top level.
    (target / "weights.csv").write_text(",kind\npresentation_score,affinity\n")
    resolved = compare_models._probe_run_dir(
        str(tmp_path / "run"), "presentation")
    assert resolved == str(target)


def test_probe_run_dir_finds_processing_root(tmp_path):
    target = tmp_path / "run" / "processing" / "models.selected.with_flanks"
    target.mkdir(parents=True)
    resolved = compare_models._probe_run_dir(
        str(tmp_path / "run"), "processing")
    assert resolved == str(tmp_path / "run" / "processing")


def test_probe_run_dir_still_finds_legacy_presentation_models_combined(tmp_path):
    target = tmp_path / "run" / "presentation" / "models.combined"
    target.mkdir(parents=True)
    (target / "weights.csv").write_text(",kind\npresentation_score,affinity\n")
    resolved = compare_models._probe_run_dir(
        str(tmp_path / "run"), "presentation")
    assert resolved == str(target)


def test_resolve_components_auto_picks_available(tmp_path):
    a_train = tmp_path / "a_train"
    a_train.mkdir()
    b_train = tmp_path / "b_train"
    b_train.mkdir()
    side_a = {
        "label": "a", "letter": "a", "spec": "a",
        "paths": {"training": str(a_train), "affinity": None,
                  "presentation": None, "processing": None},
    }
    side_b = {
        "label": "b", "letter": "b", "spec": "b",
        "paths": {"training": str(b_train), "affinity": None,
                  "presentation": None, "processing": None},
    }
    components = compare_models._resolve_components("auto", side_a, side_b)
    assert components == ["training_stats"]


def test_resolve_components_auto_includes_processing():
    side = {
        "label": "a", "letter": "a", "spec": "a",
        "paths": {"training": None, "affinity": None,
                  "presentation": None, "processing": "/tmp/processing"},
    }
    assert compare_models._resolve_components("auto", side, side) == ["processing"]


def test_resolve_components_explicit_rejects_unavailable():
    side_a = {
        "label": "a", "letter": "a", "spec": "a",
        "paths": {"training": "/tmp/x", "affinity": None,
                  "presentation": None, "processing": None},
    }
    side_b = {
        "label": "b", "letter": "b", "spec": "b",
        "paths": {"training": "/tmp/y", "affinity": None,
                  "presentation": None, "processing": None},
    }
    with pytest.raises(SystemExit, match="affinity"):
        compare_models._resolve_components(
            "affinity,training_stats", side_a, side_b)


def test_resolve_components_bad_name_raises():
    side = {"label": "a", "letter": "a", "spec": "a",
            "paths": {k: None for k in
                      ("training", "affinity", "processing", "presentation")}}
    with pytest.raises(SystemExit):
        compare_models._resolve_components("nope", side, side)


@pytest.mark.parametrize("include", ["affinity,affinity", "affinity,"])
def test_resolve_components_rejects_duplicate_or_empty_entries(include):
    side = {
        "label": "a", "letter": "a", "spec": "a",
        "paths": {"training": None, "affinity": "/tmp/affinity",
                  "processing": None, "presentation": None},
    }
    with pytest.raises(SystemExit):
        compare_models._resolve_components(include, side, side)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def test_metrics_handles_only_positives_or_only_negatives():
    only_pos = compare_models._metrics([1, 1, 1], [0.1, 0.2, 0.3])
    assert pandas.isna(only_pos["roc_auc"])
    only_neg = compare_models._metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert pandas.isna(only_neg["roc_auc"])


def test_metrics_ppv_at_n_basic():
    # 4 hits, 4 misses. Scores rank hits first → PPV@4 = 1.0
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4]
    m = compare_models._metrics(y, s)
    assert m["ppv_at_n"] == 1.0
    assert m["roc_auc"] == 1.0


def test_metrics_ignores_nans_in_scores():
    import numpy as np
    y = [1, 1, 0, 0]
    s = [0.9, np.nan, 0.1, np.nan]
    m = compare_models._metrics(y, s)
    assert m["n"] == 2


@pytest.mark.parametrize(
    "hits",
    [[], [1, 1], [0, 0]],
)
def test_comparison_requires_positive_and_negative_rows(hits):
    with pytest.raises(ValueError, match="no valid binary comparison set"):
        compare_models._require_binary_comparison_rows(
            pandas.DataFrame({"hit": hits}), "test comparison")


@pytest.mark.parametrize("hits", [[numpy.nan], [0, 1, 2]])
def test_comparison_rejects_invalid_hit_values(hits):
    with pytest.raises(ValueError, match="non-binary or missing hit"):
        compare_models._require_binary_comparison_rows(
            pandas.DataFrame({"hit": hits}), "test comparison")


def test_comparison_rejects_incomplete_benchmark_rows():
    frame = pandas.DataFrame({
        "peptide": pandas.Series(["SIINFEKL", ""], dtype="string"),
        "hla": pandas.Series(["HLA-A*02:01", None], dtype="string"),
        "hit": [1, 0],
    })
    with pytest.raises(
            ValueError,
            match=r"missing or blank required values.*peptide=1.*hla=1"):
        compare_models._require_complete_benchmark_rows(
            frame, ("peptide", "hla", "hit"), "Affinity benchmark")


def test_affinity_supported_alleles_are_normalized_and_required(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="missing its allele sequence registry"):
        compare_models._read_supported_alleles(str(model_dir))

    pandas.DataFrame({
        "allele": ["A0201", "HLA-B*07:02", "BoLA-NC*01:01"],
        "sequence": ["A" * 34, "B" * 34, "C" * 34],
    }).to_csv(model_dir / "allele_sequences.csv", index=False)
    assert compare_models._read_supported_alleles(str(model_dir)) == {
        "HLA-A*02:01", "HLA-B*07:02",
    }


def test_affinity_per_allele_excludes_single_class_groups():
    frame = pandas.DataFrame({
        "hla": ["HLA-A*02:01"] * 30 + ["HLA-B*07:02"] * 30,
        "hit": [1] * 30 + [0, 1] * 15,
        "a_score": numpy.linspace(0, 1, 60),
        "b_score": numpy.linspace(1, 0, 60),
    })
    result = compare_models._affinity_per_allele(frame)
    assert result.allele.tolist() == ["HLA-B*07:02"]


@pytest.mark.parametrize(
    "modes",
    ["with_flanks,with_flanks", "with_flanks,"],
)
def test_requested_modes_rejects_duplicate_or_empty_entries(modes):
    with pytest.raises(SystemExit):
        compare_models._requested_modes(
            modes,
            compare_models.PROCESSING_MODES,
            "--processing-modes",
        )


def test_compare_models_pct_change_is_numeric():
    assert compare_models._pct_change(0.75, 0.50) == 50.0
    assert numpy.isclose(compare_models._pct_change(0.25, 0.20), 25.0)
    assert pandas.isna(compare_models._pct_change(0.25, 0.0))
    assert pandas.isna(compare_models._pct_change(0.25, numpy.nan))


def test_processing_metrics_use_shared_non_nan_rows():
    scored = pandas.DataFrame({
        "sample_id": ["s1"] * 6,
        "hla": ["HLA-A*02:01"] * 6,
        "peptide_len": [8, 8, 9, 9, 10, 10],
        "hit": [1, 0, 1, 0, 1, 0],
        "a_processing_score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        "b_processing_score": [0.8, 0.2, 0.7, 0.3, numpy.nan, numpy.nan],
    })

    shared_scored = compare_models._shared_score_rows(scored, "processing_score")
    per_sample = compare_models._presentation_per_sample(
        shared_scored, "processing_score")
    per_length, _per_length_per_sample = compare_models._presentation_per_length(
        shared_scored, "processing_score")
    summary = compare_models._presentation_mode_summary(
        shared_scored, per_sample, per_length, "with_flanks", "processing_score")

    assert per_sample.iloc[0]["n"] == 4
    assert per_sample.iloc[0]["n_pos"] == 2
    assert summary["n_rows"] == 4
    assert summary["n_hits"] == 2


def test_processing_comparison_rejects_non_finite_scores():
    scored = pandas.DataFrame({
        "peptide": ["SIINFEKL", "AAAAAAAAA"],
        "a_processing_score": [0.9, numpy.nan],
        "b_processing_score": [0.8, 0.2],
    })

    with pytest.raises(
            ValueError,
            match=(
                r"requires every benchmark peptide.*A \(candidate\): 1 "
                r"non-finite score"
            )):
        compare_models._require_finite_processing_scores(
            scored,
            mode="with_flanks",
            labels=("candidate", "baseline"),
        )


def test_affinity_comparison_rejects_invalid_predictions():
    scored = pandas.DataFrame({
        "peptide": ["SIINFEKL", "AAAAAAAAA", "SLYNTVATL"],
        "hla": ["HLA-A*02:01"] * 3,
        "a_pred": [20.0, numpy.nan, 0.0],
        "b_pred": [30.0, 40.0, 50.0],
    })

    with pytest.raises(
            ValueError,
            match=(
                r"finite, positive IC50.*A \(candidate\): 2 invalid IC50"
            )):
        compare_models._require_valid_affinity_predictions(
            scored, labels=("candidate", "baseline"))


def test_presentation_comparison_rejects_non_finite_scores():
    scored = pandas.DataFrame({
        "peptide": ["SIINFEKL", "AAAAAAAAA"],
        "a_presentation_score": [0.9, numpy.nan],
        "b_presentation_score": [0.8, 0.2],
        "a_presentation_percentile": [1.0, 2.0],
        "b_presentation_percentile": [1.5, 2.5],
    })

    with pytest.raises(
            ValueError,
            match=(
                r"requires every benchmark row.*A \(candidate\) "
                r"presentation_score: 1 non-finite score"
            )):
        compare_models._require_finite_presentation_scores(
            scored,
            mode="with_flanks",
            labels=("candidate", "baseline"),
        )


def test_comparison_elastic_hints_follow_effective_batch_default(
        monkeypatch):
    captured = []

    def fake_worker_pool(
            args, workload_name, workload_hints, start_method=None):
        del args
        assert start_method == "spawn"
        captured.append((workload_name, dict(workload_hints)))
        return None

    args = argparse.Namespace(
        backend="cpu",
        gpus=0,
        max_workers_per_gpu=1,
        num_jobs=1,
    )
    benchmark = pandas.DataFrame({
        "peptide": ["SIINFEKL"],
        "sample_id": ["sample"],
        "hla": ["HLA-A*02:01"],
        "hit": [1],
        "peptide_len": [8],
        "n_flank": ["NNN"],
        "c_flank": ["CCC"],
    })
    monkeypatch.setattr(
        compare_models,
        "worker_pool_with_gpu_assignments_from_args",
        fake_worker_pool,
    )
    monkeypatch.setattr(
        compare_models, "default_prediction_batch_is_auto", lambda: False)
    monkeypatch.setattr(
        compare_models,
        "_predict_affinity_chunk",
        lambda *_args, **_kwargs: (0, numpy.asarray([1.0])),
    )
    monkeypatch.setattr(
        compare_models,
        "_predict_processing_chunk",
        lambda *_args, **_kwargs: (0, numpy.asarray([0.5])),
    )
    monkeypatch.setattr(
        compare_models,
        "_predict_presentation_chunk",
        lambda *_args, **_kwargs: (0, pandas.DataFrame(index=[0])),
    )

    compare_models._parallel_affinity_predict(
        args, "models", ["SIINFEKL"], ["HLA-A*02:01"], model_bytes=1)
    compare_models._parallel_processing_predict(
        args, "models", benchmark, "with_flanks", "candidate",
        model_bytes=1)
    compare_models._parallel_presentation_predict(
        args, "models", benchmark, "with_flanks", "candidate",
        model_bytes=1)

    assert [
        (workload_name, hints["elastic_batch"])
        for workload_name, hints in captured
    ] == [
        (compare_models.WORKLOAD_AFFINITY_INFERENCE, False),
        (compare_models.WORKLOAD_PROCESSING_INFERENCE, True),
        (compare_models.WORKLOAD_PRESENTATION_INFERENCE, False),
    ]


def test_processing_comparison_rejects_missing_requested_mode(
        monkeypatch, tmp_path):
    side_a_root = tmp_path / "side-a-processing"
    side_b_root = tmp_path / "side-b-processing"
    for mode in compare_models.PROCESSING_MODES:
        (side_a_root / ("models.selected.%s" % mode)).mkdir(parents=True)
    for mode in ("with_flanks", "no_flank"):
        (side_b_root / ("models.selected.%s" % mode)).mkdir(parents=True)
    side_a = {
        "label": "candidate",
        "paths": {"processing": str(side_a_root)},
    }
    side_b = {
        "label": "baseline",
        "paths": {"processing": str(side_b_root)},
    }
    args = _make_args(out=str(tmp_path / "comparison"))
    monkeypatch.setattr(
        compare_models,
        "_load_presentation_benchmark",
        lambda *_args: pytest.fail("benchmark must not load"),
    )

    with pytest.raises(
            SystemExit, match=r"short_flanks: side B \(baseline\)"):
        compare_models._run_processing(side_a, side_b, args)

    assert not (tmp_path / "comparison" / "processing").exists()


@pytest.mark.parametrize(
    ("hits", "hlas", "expected_error"),
    [
        ([1, 0.5], ["HLA-A*02:01"] * 2, "non-binary or missing hit"),
        (
            [1, 0],
            ["HLA-A*02:01", "HLA-B*07:02"],
            "sample_id.*multiple HLA genotypes",
        ),
    ],
)
def test_presentation_benchmark_rejects_invalid_evaluation_rows(
        tmp_path, hits, hlas, expected_error):
    pandas.DataFrame({
        "peptide": ["SIINFEKL", "SLYNTVATL"],
        "sample_id": ["sample"] * 2,
        "hla": hlas,
        "hit": hits,
    }).to_csv(
        tmp_path / "benchmark.multiallelic.train_excluded.test.csv.bz2",
        index=False,
    )

    with pytest.raises(ValueError, match=expected_error):
        compare_models._load_presentation_benchmark(str(tmp_path), None)


def test_presentation_benchmark_normalizes_equivalent_genotypes(tmp_path):
    pandas.DataFrame({
        "peptide": ["SIINFEKL", "SLYNTVATL"],
        "sample_id": ["sample"] * 2,
        "hla": ["B0702 A0201", "HLA-A*02:01 HLA-B*07:02"],
        "hit": [1, 0],
    }).to_csv(
        tmp_path / "benchmark.multiallelic.train_excluded.test.csv.bz2",
        index=False,
    )

    result = compare_models._load_presentation_benchmark(str(tmp_path), None)

    assert result.hla.unique().tolist() == ["HLA-A*02:01 HLA-B*07:02"]


def test_presentation_benchmark_rechecks_classes_after_length_filter(tmp_path):
    pandas.DataFrame({
        "peptide": ["SIINFEKL", "TOO-LONG-AND-INVALID"],
        "sample_id": ["sample"] * 2,
        "hla": ["HLA-A*02:01"] * 2,
        "hit": [1, 0],
    }).to_csv(
        tmp_path / "benchmark.multiallelic.train_excluded.test.csv.bz2",
        index=False,
    )

    with pytest.raises(ValueError, match="after peptide-length filtering"):
        compare_models._load_presentation_benchmark(str(tmp_path), None)


def test_affinity_metrics_handle_no_reportable_alleles():
    """Sparse smoke-test inputs may leave every allele below reporting
    filters. The component should still write schema-valid empty tables.
    """
    test = pandas.DataFrame({
        "hla": ["HLA-A*02:01", "HLA-A*02:01"],
        "peptide": ["SIINFEKL", "SLYNTVATL"],
        "peptide_len": [8, 9],
        "hit": [1, 0],
        "a_score": [0.9, 0.1],
        "b_score": [0.8, 0.2],
    })

    per_allele = compare_models._affinity_per_allele(test)
    assert per_allele.empty
    assert {"allele", "n", "n_pos", "roc_auc_diff"}.issubset(
        per_allele.columns)

    per_length, per_length_per_allele = compare_models._affinity_per_length(test)
    assert list(per_length["length"]) == [8, 9]
    assert list(per_length["n_alleles_reported"]) == [0, 0]
    assert per_length_per_allele.empty
    assert {"length", "allele", "n", "n_pos", "roc_auc_diff"}.issubset(
        per_length_per_allele.columns)

    summary = compare_models._affinity_summary(test, per_allele, per_length)
    assert summary["n_rows"] == 2
    assert summary["n_alleles_reported"] == 0
    assert summary["allele_count"]["a_better_roc_auc"] == 0
    assert summary["allele_count"]["b_better_roc_auc"] == 0


def test_processing_summary_uses_processing_score():
    scored = pandas.DataFrame({
        "peptide": ["AAAAAAAA", "BBBBBBBB", "CCCCCCCC", "DDDDDDDD"],
        "sample_id": ["s1", "s1", "s2", "s2"],
        "hla": ["HLA-A*02:01"] * 4,
        "hit": [1, 0, 1, 0],
        "peptide_len": [8, 8, 8, 8],
        "a_processing_score": [0.9, 0.1, 0.8, 0.2],
        "b_processing_score": [0.6, 0.4, 0.7, 0.3],
    })
    per_sample = compare_models._presentation_per_sample(
        scored, "processing_score")
    per_length, _ = compare_models._presentation_per_length(
        scored, "processing_score")
    summary = compare_models._presentation_mode_summary(
        scored, per_sample, per_length, "no_flank", "processing_score")
    row = compare_models._presentation_summary_row(summary)
    assert row["mode"] == "no_flank"
    assert row["a_macro_roc_auc"] == pytest.approx(1.0)
    assert row["b_macro_roc_auc"] == pytest.approx(1.0)


def test_release_summary_rows_include_affinity_processing_presentation():
    metric_summary = {
        "n_rows": 4,
        "n_hits": 2,
        "n_samples_reported": 2,
        "micro_pooled": {
            "a": {"roc_auc": 0.9, "pr_auc": 0.8, "ppv_at_n": 0.7},
            "b": {"roc_auc": 0.8, "pr_auc": 0.7, "ppv_at_n": 0.6},
        },
        "macro_mean_over_samples": {
            "roc_auc": {"a": 0.91, "b": 0.81},
            "pr_auc": {"a": 0.82, "b": 0.72},
            "ppv_at_n": {"a": 0.73, "b": 0.63},
        },
    }
    headline = {
        "affinity": {
            "micro_pooled": {
                "a": {"roc_auc": 0.98, "pr_auc": 0.68, "ppv_at_n": 0.66},
                "b": {"roc_auc": 0.97, "pr_auc": 0.65, "ppv_at_n": 0.64},
            },
            "macro_mean_over_alleles": {
                "roc_auc": {"a": 0.981, "b": 0.973},
                "pr_auc": {"a": 0.683, "b": 0.657},
                "ppv_at_n": {"a": 0.669, "b": 0.645},
            },
        },
        "processing": {
            "modes": ["no_flank"],
            "summaries": {"no_flank": {"processing_score": metric_summary}},
        },
        "presentation": {
            "modes": ["with_flanks"],
            "summaries": {"with_flanks": {"presentation_score": metric_summary}},
        },
    }
    rows = compare_models._release_summary_rows(
        headline, ["affinity", "processing", "presentation"])
    assert {row["component"] for row in rows} == {
        "affinity", "processing", "presentation"}
    assert any(row["metric"] == "AUROC" and row["average"] == "Macro"
               for row in rows)


# ---------------------------------------------------------------------------
# training_stats end-to-end
# ---------------------------------------------------------------------------


def _write_synthetic_manifest(target_dir, model_name, wall_time_sec,
                              n_finetune_epochs):
    target_dir.mkdir(parents=True, exist_ok=True)
    fit_info = [{
        "training_info": {"phase": "finetune", "fold_num": 0},
        "time": wall_time_sec,
        "loss": [0.5] * n_finetune_epochs,
        "val_loss": [0.6] * n_finetune_epochs,
    }]
    config_json = json.dumps({
        "hyperparameters": {"layer_sizes": [32]},
        "fit_info": fit_info,
    })
    df = pandas.DataFrame([{
        "model_name": model_name, "config_json": config_json,
    }])
    df.to_csv(target_dir / "manifest.csv", index=False)


def test_training_stats_component_end_to_end(tmp_path):
    a_dir = tmp_path / "run_a" / "models.unselected.combined"
    _write_synthetic_manifest(a_dir, "model_a", wall_time_sec=120.0,
                              n_finetune_epochs=10)
    b_dir = tmp_path / "run_b" / "models.unselected.combined"
    _write_synthetic_manifest(b_dir, "model_b", wall_time_sec=60.0,
                              n_finetune_epochs=20)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    side_a = compare_models._resolve_side(
        "a", str(tmp_path / "run_a"), "a", _make_args())
    side_b = compare_models._resolve_side(
        "b", str(tmp_path / "run_b"), "b", _make_args())
    assert side_a["paths"]["training"].endswith("models.unselected.combined")

    headline = compare_models._run_training_stats(
        side_a, side_b, str(out_dir))
    per_task = pandas.read_csv(out_dir / "training_stats" / "per_task.csv")
    summary = pandas.read_csv(out_dir / "training_stats" / "summary.csv")
    assert set(per_task["side"]) == {"a", "b"}
    assert set(summary["side"]) == {"a", "b"}
    # Side A wall-time was double B's.
    assert headline["side_a_finetune_total_wall_min"] == pytest.approx(2.0)
    assert headline["side_b_finetune_total_wall_min"] == pytest.approx(1.0)


def test_training_stats_handles_colliding_labels(tmp_path):
    """If --a-label and --b-label collide, positional indexing keeps the
    headline pointing at the correct side."""
    a_dir = tmp_path / "run_a" / "models.unselected.combined"
    _write_synthetic_manifest(a_dir, "model_a", wall_time_sec=120.0,
                              n_finetune_epochs=10)
    b_dir = tmp_path / "run_b" / "models.unselected.combined"
    _write_synthetic_manifest(b_dir, "model_b", wall_time_sec=60.0,
                              n_finetune_epochs=20)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    side_a = compare_models._resolve_side(
        "a", str(tmp_path / "run_a"), "collision", _make_args())
    side_b = compare_models._resolve_side(
        "b", str(tmp_path / "run_b"), "collision", _make_args())
    headline = compare_models._run_training_stats(
        side_a, side_b, str(out_dir))
    # 2.0 vs 1.0 must map to A vs B by position, not by label lookup.
    assert headline["side_a_finetune_total_wall_min"] == pytest.approx(2.0)
    assert headline["side_b_finetune_total_wall_min"] == pytest.approx(1.0)


def test_load_training_summary_rejects_bad_manifest_schema(tmp_path):
    """A manifest missing required columns should fail loudly with the
    missing column names + the manifest path, not AttributeError."""
    bad_dir = tmp_path / "run"
    bad_dir.mkdir()
    # Manifest is missing 'config_json'.
    (bad_dir / "manifest.csv").write_text("model_name,something_else\nx,1\n")
    with pytest.raises(ValueError, match="config_json"):
        compare_models._load_training_summary(str(bad_dir))


def test_run_orchestrator_training_stats_only(tmp_path):
    """End-to-end smoke for run(): training_stats only, both sides
    synthetic. Catches regressions in is_public, _stamp, and the
    headline-by-label bugs all at once."""
    for letter in ("a", "b"):
        target = tmp_path / ("run_" + letter) / "models.unselected.combined"
        _write_synthetic_manifest(
            target, "model_" + letter,
            wall_time_sec=180.0 if letter == "a" else 90.0,
            n_finetune_epochs=10,
        )
    out_dir = tmp_path / "out"
    args = _make_args(
        a=str(tmp_path / "run_a"),
        b=str(tmp_path / "run_b"),
        a_label="candidate",
        b_label="baseline",
        out=str(out_dir),
        include="training_stats",
    )
    assert compare_models.run(args) == 0
    # Side files written.
    assert json.loads((out_dir / "side_a.json").read_text())["label"] == "candidate"
    assert json.loads((out_dir / "side_b.json").read_text())["label"] == "baseline"
    # Top-level summary mentions both labels.
    summary_md = (out_dir / "summary.md").read_text()
    assert "candidate" in summary_md
    assert "baseline" in summary_md
    assert "training_stats" in summary_md
    # Component CSVs landed.
    per_task = pandas.read_csv(out_dir / "training_stats" / "per_task.csv")
    assert set(per_task["side"]) == {"candidate", "baseline"}


# ---------------------------------------------------------------------------
# plot_model_comparison detection
# ---------------------------------------------------------------------------


def test_detect_available_components_empty(tmp_path):
    assert plot_model_comparison._detect_available_components(str(tmp_path)) == []


def test_compare_models_reset_removes_only_owned_outputs(tmp_path):
    for name in ("affinity", "processing", "plots", "worker_logs"):
        path = tmp_path / name
        path.mkdir()
        (path / "stale.txt").write_text("stale")
    (tmp_path / "release_summary.csv").write_text("stale")
    (tmp_path / "summary.pdf").write_bytes(b"stale")
    (tmp_path / "keep.txt").write_text("user-owned")

    compare_models._reset_comparison_outputs(str(tmp_path))

    assert not (tmp_path / "affinity").exists()
    assert not (tmp_path / "processing").exists()
    assert not (tmp_path / "plots").exists()
    assert not (tmp_path / "worker_logs").exists()
    assert not (tmp_path / "release_summary.csv").exists()
    assert not (tmp_path / "summary.pdf").exists()
    assert (tmp_path / "keep.txt").read_text() == "user-owned"


def test_compare_models_invalid_component_preserves_previous_outputs(tmp_path):
    for letter in ("a", "b"):
        target = tmp_path / ("run_" + letter) / "models.unselected.combined"
        _write_synthetic_manifest(
            target, "model_" + letter,
            wall_time_sec=10.0, n_finetune_epochs=1)
    out_dir = tmp_path / "out"
    stale = out_dir / "affinity" / "predictions.csv.bz2"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"previous comparison")
    args = _make_args(
        a=str(tmp_path / "run_a"),
        b=str(tmp_path / "run_b"),
        out=str(out_dir),
        include="training_stats,affinity",
    )

    with pytest.raises(SystemExit, match="affinity"):
        compare_models.run(args)

    assert stale.read_bytes() == b"previous comparison"


def test_compare_models_rejects_output_that_contains_model_input(tmp_path):
    args = compare_models.make_parser().parse_args([
        "--a", str(tmp_path),
        "--b", "public:2.2.0",
        "--out", str(tmp_path),
    ])

    with pytest.raises(ValueError, match="cannot contain an input path"):
        compare_models._validate_comparison_output_location(args)


def test_compare_models_rejects_output_that_contains_benchmark_input(tmp_path):
    data_dir = tmp_path / "processing" / "benchmarks"
    data_dir.mkdir(parents=True)
    args = compare_models.make_parser().parse_args([
        "--a", "public:2.3.0",
        "--b", "public:2.2.0",
        "--data-dir", str(data_dir),
        "--out", str(tmp_path),
    ])

    with pytest.raises(ValueError, match="--data-dir"):
        compare_models._validate_comparison_output_location(args)


def test_compare_models_side_json_records_model_provenance(tmp_path):
    paths = {}
    for role, subdir in (
            ("affinity", "affinity/models.combined"),
            ("presentation", "presentation/models")):
        path = tmp_path / subdir
        path.mkdir(parents=True)
        (path / "info.txt").write_text("package\tmhcflurry 2.3.0rc14\n")
        paths[role] = str(path)
    processing = tmp_path / "processing" / "models.selected.with_flanks"
    processing.mkdir(parents=True)
    (processing / "info.txt").write_text("package mhcflurry 2.3.0rc14\n")
    paths["processing"] = str(tmp_path / "processing")
    paths["training"] = None
    provenance = {"release": "2.3.0", "workflow_id": "run-123"}
    (tmp_path / "release_provenance.json").write_text(json.dumps(provenance))

    result = compare_models._side_to_json({
        "letter": "a",
        "spec": str(tmp_path),
        "label": "candidate",
        "paths": paths,
    })

    assert result["model_package_versions"] == {
        "affinity": ["2.3.0rc14"],
        "presentation": ["2.3.0rc14"],
        "processing": ["2.3.0rc14"],
    }
    assert result["release_provenance"] == provenance


def test_plot_model_comparison_rejects_missing_component_without_cleanup(
        tmp_path):
    plot_dir = tmp_path / "plots"
    paper_dir = plot_dir / "paper_figures"
    paper_dir.mkdir(parents=True)
    (plot_dir / "stale.pdf").write_bytes(b"stale")
    (paper_dir / "paper_figures.pdf").write_bytes(b"paper")
    summary = plot_dir / "summary.pdf"
    summary.write_bytes(b"summary")
    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(tmp_path),
        "--components", "affinity",
        "--summary-pdf", str(summary),
    ])

    with pytest.raises(SystemExit, match="affinity"):
        plot_model_comparison.run(args)

    assert (plot_dir / "stale.pdf").read_bytes() == b"stale"
    assert (paper_dir / "paper_figures.pdf").read_bytes() == b"paper"
    assert summary.read_bytes() == b"summary"


@pytest.mark.parametrize(("option", "relative_path", "is_directory"), [
    ("--paper-figures-scores-dir", "saved_scores", True),
    (
        "--paper-figures-multiallelic-predictions",
        "saved_predictions/multiallelic.csv",
        False,
    ),
    (
        "--paper-figures-monoallelic-predictions",
        "saved_predictions/monoallelic.csv",
        False,
    ),
])
def test_plot_model_comparison_rejects_inputs_under_cleanup_tree(
        tmp_path, monkeypatch, option, relative_path, is_directory):
    plot_dir = tmp_path / "plots"
    input_path = plot_dir / relative_path
    if is_directory:
        input_path.mkdir(parents=True)
        sentinel = input_path / "input.csv"
    else:
        input_path.parent.mkdir(parents=True)
        sentinel = input_path
    sentinel.write_bytes(b"paper input")
    stale_plot = plot_dir / "stale.pdf"
    stale_plot.write_bytes(b"stale plot")
    monkeypatch.setattr(
        plot_model_comparison, "_apply_paper_style", lambda: None)
    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(tmp_path),
        option, str(input_path),
    ])

    with pytest.raises(SystemExit, match="Refusing to delete input"):
        plot_model_comparison.run(args)

    assert sentinel.read_bytes() == b"paper input"
    assert stale_plot.read_bytes() == b"stale plot"


def test_plot_model_comparison_rejects_symlink_to_input_under_cleanup_tree(
        tmp_path, monkeypatch):
    scores_dir = tmp_path / "plots" / "saved_scores"
    scores_dir.mkdir(parents=True)
    sentinel = scores_dir / "input.csv"
    sentinel.write_bytes(b"paper input")
    scores_link = tmp_path / "scores-link"
    try:
        scores_link.symlink_to(scores_dir, target_is_directory=True)
    except OSError as error:
        pytest.skip("symlinks unavailable: %s" % error)
    monkeypatch.setattr(
        plot_model_comparison, "_apply_paper_style", lambda: None)
    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(tmp_path),
        "--paper-figures-scores-dir", str(scores_link),
    ])

    with pytest.raises(SystemExit, match="Refusing to delete input"):
        plot_model_comparison.run(args)

    assert sentinel.read_bytes() == b"paper input"


def test_plot_model_comparison_rejects_paper_output_equal_to_plot_directory(
        tmp_path, monkeypatch):
    plot_dir = tmp_path / "plots"
    combined = plot_dir / "paper_figures.pdf"
    panel = plot_dir / "pdf" / "panel.pdf"
    panel.parent.mkdir(parents=True)
    combined.write_bytes(b"combined paper figures")
    panel.write_bytes(b"paper panel")
    monkeypatch.setattr(
        plot_model_comparison, "_apply_paper_style", lambda: None)
    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(tmp_path),
        "--paper-figures-out", str(plot_dir),
    ])

    with pytest.raises(SystemExit, match="dedicated directory"):
        plot_model_comparison.run(args)

    assert combined.read_bytes() == b"combined paper figures"
    assert panel.read_bytes() == b"paper panel"


@pytest.mark.parametrize(
    ("option", "relative_path"),
    [
        ("--summary-pdf", "release_summary.csv"),
        ("--paper-figures-out", "affinity"),
    ],
)
def test_plot_model_comparison_rejects_outputs_inside_comparison_inputs(
        tmp_path, monkeypatch, option, relative_path):
    comparison_dir = tmp_path / "comparison"
    target = comparison_dir / relative_path
    if option == "--summary-pdf":
        target.parent.mkdir(parents=True)
        target.write_bytes(b"comparison input")
    else:
        target.mkdir(parents=True)
        (target / "predictions.csv.bz2").write_bytes(b"comparison input")
    monkeypatch.setattr(
        plot_model_comparison, "_apply_paper_style", lambda: None)
    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(comparison_dir),
        option, str(target),
    ])

    with pytest.raises(SystemExit, match="comparison input tree"):
        plot_model_comparison.run(args)

    if target.is_file():
        assert target.read_bytes() == b"comparison input"
    else:
        assert (target / "predictions.csv.bz2").read_bytes() == b"comparison input"


@pytest.mark.parametrize(("owner", "relative_path"), [
    ("paper", "paper_figures.pdf"),
    ("plots", "affinity/roc.pdf"),
    ("plots", "paper/release_summary_macro.pdf"),
])
def test_plot_model_comparison_rejects_summary_pdf_output_collisions(
        tmp_path, monkeypatch, owner, relative_path):
    comparison_dir = tmp_path / "comparison"
    plot_dir = comparison_dir / "plots"
    paper_dir = tmp_path / "external-paper"
    summary = (
        paper_dir if owner == "paper" else plot_dir
    ) / relative_path
    summary.parent.mkdir(parents=True)
    summary.write_bytes(b"generated figure")
    stale_plot = plot_dir / "stale.pdf"
    stale_plot.parent.mkdir(parents=True, exist_ok=True)
    stale_plot.write_bytes(b"stale plot")
    monkeypatch.setattr(
        plot_model_comparison, "_apply_paper_style", lambda: None)
    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(comparison_dir),
        "--paper-figures-out", str(paper_dir),
        "--summary-pdf", str(summary),
    ])

    with pytest.raises(SystemExit, match="collides with command-owned"):
        plot_model_comparison.run(args)

    assert summary.read_bytes() == b"generated figure"
    assert stale_plot.read_bytes() == b"stale plot"


def test_detect_available_components_finds_affinity(tmp_path):
    aff = tmp_path / "affinity"
    aff.mkdir()
    (aff / "predictions.csv.bz2").write_text("hit,a_score,b_score\n")
    assert "affinity" in plot_model_comparison._detect_available_components(
        str(tmp_path))


def test_detect_available_components_finds_processing(tmp_path):
    (tmp_path / "processing").mkdir()
    assert "processing" in plot_model_comparison._detect_available_components(
        str(tmp_path))


def test_read_optional_csv_tolerates_empty_summary(tmp_path):
    path = tmp_path / "summary_table.csv"
    path.write_text("\n")
    summary = plot_model_comparison._read_optional_csv(str(path))
    assert summary.empty


def test_detect_available_components_finds_presentation(tmp_path):
    (tmp_path / "presentation").mkdir()
    assert "presentation" in plot_model_comparison._detect_available_components(
        str(tmp_path))


@pytest.mark.parametrize("components", ["affinity,affinity", "affinity,"])
def test_plot_model_comparison_rejects_duplicate_or_empty_components(
        tmp_path, monkeypatch, components):
    affinity = tmp_path / "affinity"
    affinity.mkdir()
    (affinity / "summary.json").write_text("{}")
    monkeypatch.setattr(
        plot_model_comparison, "_apply_paper_style", lambda: None)
    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(tmp_path),
        "--components", components,
    ])

    with pytest.raises(SystemExit):
        plot_model_comparison.run(args)


def test_load_side_labels_falls_back_when_missing(tmp_path):
    labels = plot_model_comparison._load_side_labels(str(tmp_path))
    assert labels == {"a": "Side A", "b": "Side B"}


def test_load_side_labels_reads_json(tmp_path):
    (tmp_path / "side_a.json").write_text(json.dumps({"label": "candidate"}))
    (tmp_path / "side_b.json").write_text(json.dumps({"label": "baseline"}))
    labels = plot_model_comparison._load_side_labels(str(tmp_path))
    assert labels == {"a": "candidate", "b": "baseline"}


def test_roc_pr_plots_skip_single_class_slices(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    y = numpy.asarray([1, 1, 1])
    a_score = numpy.asarray([0.1, 0.2, 0.3])
    b_score = numpy.asarray([numpy.nan, numpy.nan, numpy.nan])

    roc_path = tmp_path / "roc.png"
    pr_path = tmp_path / "pr.png"
    plot_model_comparison._save_roc(
        plt, roc_curve, roc_auc_score,
        y, a_score, b_score, "a", "b", str(roc_path), "ROC")
    plot_model_comparison._save_pr(
        plt, precision_recall_curve, average_precision_score,
        y, a_score, b_score, "a", "b", str(pr_path), "PR")
    assert roc_path.is_file()
    assert pr_path.is_file()


def test_comparison_curves_use_shared_finite_rows():
    y, a_score, b_score = plot_model_comparison._shared_finite_curve_values(
        y=[1, 0, 1, 0],
        a_score=[0.9, numpy.nan, 0.8, 0.1],
        b_score=[numpy.nan, 0.2, 0.7, 0.3],
    )

    assert y.tolist() == [1, 0]
    assert a_score.tolist() == [0.8, 0.1]
    assert b_score.tolist() == [0.7, 0.3]


def test_plot_model_comparison_writes_paper_plots_from_summaries(tmp_path):
    pytest.importorskip("matplotlib")

    (tmp_path / "affinity").mkdir()
    (tmp_path / "processing").mkdir()
    (tmp_path / "presentation").mkdir()
    (tmp_path / "side_a.json").write_text(json.dumps({"label": "new"}))
    (tmp_path / "side_b.json").write_text(json.dumps({"label": "public"}))

    pandas.DataFrame([
        {
            "component": "affinity",
            "eval": "affinity",
            "metric": metric,
            "average": "Macro",
            "side_a": 0.8,
            "side_b": 0.7,
            "diff": 0.1,
            "pct_change": 10.0,
            "flank_mode": "",
        }
        for metric in ("AUROC", "AUPRC", "PPV@N")
    ]).to_csv(tmp_path / "release_summary.csv", index=False)

    pandas.DataFrame([
        {
            "allele": "HLA-A*02:01",
            "n": 10,
            "n_pos": 2,
            "a_roc_auc": 0.90,
            "b_roc_auc": 0.85,
            "roc_auc_diff": 0.05,
            "a_pr_auc": 0.50,
            "b_pr_auc": 0.45,
            "pr_auc_diff": 0.05,
            "a_ppv_at_n": 0.40,
            "b_ppv_at_n": 0.30,
            "ppv_at_n_diff": 0.10,
        },
        {
            "allele": "HLA-B*07:02",
            "n": 10,
            "n_pos": 2,
            "a_roc_auc": 0.80,
            "b_roc_auc": 0.75,
            "roc_auc_diff": 0.05,
            "a_pr_auc": 0.40,
            "b_pr_auc": 0.35,
            "pr_auc_diff": 0.05,
            "a_ppv_at_n": 0.30,
            "b_ppv_at_n": 0.20,
            "ppv_at_n_diff": 0.10,
        },
    ]).to_csv(tmp_path / "affinity" / "per_allele.csv", index=False)

    pandas.DataFrame([
        {
            "length": 8,
            "a_macro_roc_auc": 0.90,
            "b_macro_roc_auc": 0.80,
            "a_macro_pr_auc": 0.50,
            "b_macro_pr_auc": 0.40,
            "a_macro_ppv_at_n": 0.30,
            "b_macro_ppv_at_n": 0.20,
        },
        {
            "length": 9,
            "a_macro_roc_auc": 0.91,
            "b_macro_roc_auc": 0.81,
            "a_macro_pr_auc": 0.51,
            "b_macro_pr_auc": 0.41,
            "a_macro_ppv_at_n": 0.31,
            "b_macro_ppv_at_n": 0.21,
        },
    ]).to_csv(tmp_path / "affinity" / "per_length.csv", index=False)

    component_rows = [
        {
            "sample_id": sample_id,
            "a_roc_auc": 0.90 + offset,
            "b_roc_auc": 0.85,
            "a_pr_auc": 0.50 + offset,
            "b_pr_auc": 0.45,
            "a_ppv_at_n": 0.40 + offset,
            "b_ppv_at_n": 0.35,
        }
        for sample_id, offset in (("s1", 0.00), ("s2", 0.01))
    ]
    length_rows = [
        {
            "length": length,
            "a_macro_roc_auc": 0.90,
            "b_macro_roc_auc": 0.85,
            "a_macro_pr_auc": 0.50,
            "b_macro_pr_auc": 0.45,
            "a_macro_ppv_at_n": 0.40,
            "b_macro_ppv_at_n": 0.35,
        }
        for length in (8, 9)
    ]
    for component, mode, score_kind in (
            ("processing", "no_flank", "processing_score"),
            ("presentation", "with_flanks", "presentation_score")):
        pandas.DataFrame(component_rows).to_csv(
            tmp_path / component / (
                "per_sample_%s_%s.csv" % (mode, score_kind)),
            index=False,
        )
        pandas.DataFrame(length_rows).to_csv(
            tmp_path / component / (
                "per_length_%s_%s.csv" % (mode, score_kind)),
            index=False,
        )

    args = plot_model_comparison.make_parser().parse_args([
        "--input", str(tmp_path),
        "--a-label", "MHCflurry 2.3.0",
        "--b-label", "MHCflurry 2.2.0",
        "--summary-pdf", str(tmp_path / "plots" / "figures.pdf"),
    ])
    assert plot_model_comparison.run(args) == 0

    for path in [
        tmp_path / "plots" / "paper" / "release_summary_macro.png",
        tmp_path / "plots" / "paper" / "affinity_per_allele_scatter.png",
        tmp_path / "plots" / "paper" / "affinity_per_length_macro.png",
        tmp_path / "plots" / "paper" / "processing_per_sample_scatter.png",
        tmp_path / "plots" / "paper" / "presentation_per_length_macro.png",
        tmp_path / "plots" / "paper" / "presentation_per_length_macro.pdf",
        tmp_path / "plots" / "figures.pdf",
    ]:
        assert path.is_file()


def test_summary_pdf_falls_back_from_corrupt_pdf(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = tmp_path / "plots"
    paper_dir = plot_dir / "paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "bad.pdf").write_bytes(b"not a pdf")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(paper_dir / "fallback.png")
    plt.close(fig)

    out = plot_dir / "summary.pdf"
    plot_model_comparison._write_summary_pdf(plot_dir, out)
    assert out.is_file()


def test_summary_pdf_png_fallback_excludes_paper_figures_by_default(tmp_path):
    plot_dir = tmp_path / "plots"
    paper_png = plot_dir / "paper_figures" / "panel.png"
    diagnostic_png = plot_dir / "affinity" / "panel.png"
    paper_png.parent.mkdir(parents=True)
    diagnostic_png.parent.mkdir(parents=True)
    paper_png.write_bytes(b"unused")
    diagnostic_png.write_bytes(b"unused")
    out = plot_dir / "summary.pdf"

    assert not plot_model_comparison._include_png_in_summary(
        paper_png, plot_dir, out, include_paper_figures=False)
    assert plot_model_comparison._include_png_in_summary(
        paper_png, plot_dir, out, include_paper_figures=True)
    assert plot_model_comparison._include_png_in_summary(
        diagnostic_png, plot_dir, out, include_paper_figures=False)


def test_summary_pdf_uses_actual_paper_figures_dir(tmp_path):
    plot_dir = tmp_path / "plots"
    custom_dir = plot_dir / "custom_paper"
    combined = custom_dir / "paper_figures.pdf"
    individual = custom_dir / "pdf" / "panel.pdf"
    legacy_individual = plot_dir / "paper_2023" / "pdf" / "panel.pdf"
    diagnostic = plot_dir / "aaa" / "diagnostic.pdf"
    for path in [combined, individual, legacy_individual, diagnostic]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unused")
    out = plot_dir / "summary.pdf"

    paths = plot_model_comparison._summary_pdf_paths(
        plot_dir,
        out,
        include_paper_figures=True,
        paper_figures_dir=custom_dir,
    )

    assert paths == [combined, diagnostic]
    assert not plot_model_comparison._include_pdf_in_summary(
        individual,
        plot_dir,
        out,
        include_paper_figures=True,
        paper_figures_dir=custom_dir,
    )
    assert not plot_model_comparison._include_pdf_in_summary(
        legacy_individual,
        plot_dir,
        out,
        include_paper_figures=True,
        paper_figures_dir=custom_dir,
    )
    assert not plot_model_comparison._include_pdf_in_summary(
        combined,
        plot_dir,
        out,
        include_paper_figures=False,
        paper_figures_dir=custom_dir,
    )


def test_summary_pdf_can_include_external_paper_figures_dir(tmp_path):
    plot_dir = tmp_path / "plots"
    external_dir = tmp_path / "external_paper"
    combined = external_dir / "paper_figures.pdf"
    diagnostic = plot_dir / "paper" / "diagnostic.pdf"
    combined.parent.mkdir(parents=True)
    diagnostic.parent.mkdir(parents=True)
    combined.write_bytes(b"unused")
    diagnostic.write_bytes(b"unused")
    out = plot_dir / "summary.pdf"

    paths = plot_model_comparison._summary_pdf_paths(
        plot_dir,
        out,
        include_paper_figures=True,
        paper_figures_dir=external_dir,
    )

    assert paths == [combined, diagnostic]


def test_paper_figures_orients_configured_scores():
    score = numpy.array([0.1, 2.0, 50.0])

    for predictor in [
            "mhcflurry_production",
            "mhcflurry_production_affinity",
            "netmhcpan4.ba_affinity",
            "netmhcpan4.el_rank"]:
        assert numpy.allclose(
            paper_figures._orient_prediction_score(predictor, score),
            -score)
    assert numpy.allclose(
        paper_figures._orient_prediction_score("mixmhcpred", score),
        score)
    for predictor in ["netmhcpan4.el", "netmhcpan4.2.el"]:
        assert numpy.allclose(
            paper_figures._orient_prediction_score(predictor, score),
            score)
    assert numpy.allclose(
        paper_figures._orient_prediction_score(
            "custom_rank", score, {"custom_rank": True}),
        score)
    with pytest.raises(ValueError, match="No score orientation configured"):
        paper_figures._orient_prediction_score("custom_rank", score)


def test_paper_figures_score_predictions_uses_explicit_orientation(tmp_path):
    predictions = tmp_path / "predictions.csv"
    pandas.DataFrame([
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAK",
            "hit": 1,
            "mhcflurry_production": 20.0,
            "netmhcpan4.ba_affinity": 40.0,
            "netmhcpan4.2.el": 0.9,
        },
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAL",
            "hit": 0,
            "mhcflurry_production": 2000.0,
            "netmhcpan4.ba_affinity": 4000.0,
            "netmhcpan4.2.el": 0.1,
        },
    ]).to_csv(predictions, index=False)

    scores = paper_figures.score_saved_prediction_table(
        predictions,
        kind="multiallelic",
        external_baselines=(("netmhcpan4.ba", "ba"),),
    )
    all_scores = scores.loc[scores["length_label"] == "All"]

    assert all_scores.set_index("predictor").loc[
        "mhcflurry_production", "auc"] == 1.0
    assert all_scores.set_index("predictor").loc[
        "netmhcpan4.ba", "auc"] == 1.0
    assert all_scores.set_index("predictor").loc[
        "netmhcpan4.2.el", "auc"] == 1.0
    assert "percent_change_auc_ba" in scores.columns


def test_paper_figures_score_predictions_uses_schema_not_numeric_dtype(tmp_path):
    predictions = tmp_path / "predictions.csv"
    pandas.DataFrame([
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAK",
            "peptide_num": 0,
            "hit": 1,
            "presentation_score": 0.9,
        },
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAL",
            "peptide_num": 1,
            "hit": 0,
            "presentation_score": 0.1,
        },
    ]).to_csv(predictions, index=False)

    scores = paper_figures.score_saved_prediction_table(
        predictions,
        kind="multiallelic",
        external_baselines=(),
    )

    assert set(scores["predictor"]) == {"presentation_score"}
    assert scores.loc[scores["length_label"] == "All", "auc"].iloc[0] == 1.0


def test_paper_figures_score_predictions_requires_recognized_or_explicit_score(
        tmp_path):
    predictions = tmp_path / "predictions.csv"
    pandas.DataFrame({
        "sample_id": ["sample1", "sample1"],
        "hit": [1, 0],
        "peptide_num": [0, 1],
    }).to_csv(predictions, index=False)

    with pytest.raises(ValueError, match="no recognized predictor score columns"):
        paper_figures.score_saved_prediction_table(
            predictions,
            kind="multiallelic",
            external_baselines=(),
        )


def test_paper_figures_score_predictions_rejects_nonbinary_labels(tmp_path):
    predictions = tmp_path / "predictions.csv"
    pandas.DataFrame({
        "sample_id": ["sample1", "sample1"],
        "hit": [1, 0.5],
        "presentation_score": [0.9, 0.1],
    }).to_csv(predictions, index=False)

    with pytest.raises(ValueError, match="non-binary or missing hit"):
        paper_figures.score_saved_prediction_table(
            predictions,
            kind="multiallelic",
            external_baselines=(),
        )


def test_paper_figures_score_predictions_uses_range_index_predictor_info(
        tmp_path):
    predictions = tmp_path / "predictions.csv"
    pandas.DataFrame([
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAK",
            "hit": 1,
            "custom_model": 0.9,
        },
        {
            "sample_id": "sample1",
            "peptide": "AAAAAAAAL",
            "hit": 0,
            "custom_model": 0.1,
        },
    ]).to_csv(predictions, index=False)
    predictor_info = pandas.DataFrame([{
        "predictor": "custom_model",
        "higher_is_better": True,
    }])

    scores = paper_figures.score_saved_prediction_table(
        predictions,
        kind="multiallelic",
        predictor_info=predictor_info,
    )

    all_scores = scores.loc[scores["length_label"] == "All"]
    assert all_scores.set_index("predictor").loc["custom_model", "auc"] == 1.0


def test_paper_figures_percent_change_columns_are_numeric():
    scores = pandas.DataFrame([
        {
            "sample_id": "s1",
            "length_label": "All",
            "predictor": "candidate",
            "auc": 0.75,
            "ppv": 0.25,
        },
        {
            "sample_id": "s1",
            "length_label": "All",
            "predictor": "baseline",
            "auc": 0.50,
            "ppv": 0.20,
        },
    ])

    result = paper_figures._add_percent_change_columns(
        scores, external_baselines=(("baseline", "base"),))
    candidate = result.loc[result["predictor"] == "candidate"].iloc[0]

    assert candidate["percent_change_auc_base"] == 50.0
    assert numpy.isclose(candidate["percent_change_ppv_base"], 25.0)


def test_paper_figures_mean_ppv_small_uses_with_flanks_and_weighted_external():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class CaptureWriter:
        def __init__(self):
            self.fig = None
            self.skipped = False

        def save(self, fig, _name, _family, note=""):
            self.fig = fig

        def skip(self, *_args, **_kwargs):
            self.skipped = True

    predictor_info = pandas.DataFrame([
        {"predictor": "mhcflurry_production", "short": "BA", "color": "(0.1, 0.2, 0.3)"},
        {
            "predictor": "presentation_with_flanks_processing_score",
            "short": "AP +flanks",
            "color": "(0.2, 0.3, 0.4)",
        },
        {
            "predictor": "presentation_with_flanks_presentation_score",
            "short": "PS +flanks",
            "color": "(0.3, 0.4, 0.5)",
        },
        {
            "predictor": "presentation_without_flanks_processing_score",
            "short": "AP -flanks",
            "color": "(0.8, 0.1, 0.1)",
        },
        {"predictor": "netmhcpan4.ba", "short": "BA ext", "color": "(0.4, 0.4, 0.4)"},
        {"predictor": "netmhcpan4.el", "short": "EL ext", "color": "(0.5, 0.5, 0.5)"},
    ]).set_index("predictor")
    scores = pandas.DataFrame([
        {"sample_id": "s1", "length": numpy.nan, "length_label": "All", "predictor": "mhcflurry_production", "ppv": 0.60},
        {"sample_id": "s1", "length": numpy.nan, "length_label": "All", "predictor": "presentation_with_flanks_processing_score", "ppv": 0.50},
        {"sample_id": "s1", "length": numpy.nan, "length_label": "All", "predictor": "presentation_with_flanks_presentation_score", "ppv": 0.80},
        {"sample_id": "s1", "length": numpy.nan, "length_label": "All", "predictor": "presentation_without_flanks_processing_score", "ppv": 0.99},
        {"sample_id": "s1", "length": numpy.nan, "length_label": "All", "predictor": "netmhcpan4.ba", "ppv": 0.10},
        {"sample_id": "s2", "length": numpy.nan, "length_label": "All", "predictor": "netmhcpan4.ba", "ppv": 0.30},
        {"sample_id": "s1", "length": numpy.nan, "length_label": "All", "predictor": "netmhcpan4.el", "ppv": 0.90},
    ])
    writer = CaptureWriter()
    predictors = paper_figures.PredictorConfig(
        candidate="mhcflurry_production",
        external_baselines=(
            ("netmhcpan4.ba", "ba"),
            ("netmhcpan4.el", "el"),
        ),
        preferred_predictors=(),
        presentation_panel_predictors=(),
        presentation_panel_baselines=(),
    )

    paper_figures._plot_mean_ppv_small(
        scores,
        predictor_info,
        recent_sample_ids=None,
        note="",
        name="test",
        writer=writer,
        predictors=predictors,
    )

    assert not writer.skipped
    ax = writer.fig.axes[0]
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    heights = [patch.get_height() for patch in ax.patches]
    plt.close(writer.fig)

    assert labels == ["BA", "AP +flanks", "PS +flanks", "External tools"]
    assert numpy.allclose(heights, [0.60, 0.50, 0.80, 0.55])


def test_summary_pdf_png_fallback_handles_external_paper_dir(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = tmp_path / "plots"
    external_dir = tmp_path / "external_paper"
    diagnostic_dir = plot_dir / "paper"
    for path in [external_dir, diagnostic_dir]:
        path.mkdir(parents=True)
    for path in [
            external_dir / "external_panel.png",
            diagnostic_dir / "diagnostic_panel.png"]:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(path)
        plt.close(fig)

    out = plot_dir / "summary.pdf"
    plot_model_comparison._write_summary_pdf_from_pngs(
        plot_dir,
        out,
        include_paper_figures=True,
        paper_figures_dir=external_dir,
    )

    assert out.is_file()


def test_paper_figures_writes_available_2023_style_panels(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    pandas.DataFrame([
        {
            "predictor": "netmhcpan4.ba",
            "description": "NetMHCpan BA",
            "primary": True,
            "color": "(0.8, 0.2, 0.1)",
            "short": "NetMHCpan BA",
            "detail": "-",
        },
        {
            "predictor": "netmhcpan4.el",
            "description": "NetMHCpan EL",
            "primary": True,
            "color": "(1.0, 0.6, 0.6)",
            "short": "NetMHCpan EL",
            "detail": "-",
        },
        {
            "predictor": "mixmhcpred",
            "description": "MixMHCpred",
            "primary": True,
            "color": "(0.2, 0.5, 0.8)",
            "short": "MixMHCpred",
            "detail": "-",
        },
        {
            "predictor": "mhcflurry_production",
            "description": "MHCflurry BA",
            "primary": True,
            "color": "(0.5, 0.4, 0.8)",
            "short": "MHCflurry BA",
            "detail": "-",
        },
        {
            "predictor": "presentation_without_flanks_presentation_score",
            "description": "MHCflurry PS -flanks",
            "primary": True,
            "color": "(0.9, 0.6, 0.2)",
            "short": "PS -flanks",
            "detail": "-",
        },
        {
            "predictor": "presentation_with_flanks_presentation_score",
            "description": "MHCflurry PS +flanks",
            "primary": True,
            "color": "(0.3, 0.7, 0.2)",
            "short": "PS +flanks",
            "detail": "-",
        },
        {
            "predictor": "presentation_without_flanks_processing_score",
            "description": "MHCflurry AP -flanks",
            "primary": False,
            "color": "(0.1, 0.4, 0.2)",
            "short": "AP -flanks",
            "detail": "-",
        },
        {
            "predictor": "presentation_with_flanks_processing_score",
            "description": "MHCflurry AP +flanks",
            "primary": False,
            "color": "(0.8, 0.5, 0.7)",
            "short": "AP +flanks",
            "detail": "-",
        },
    ]).to_csv(artifacts / "predictor_info.csv", index=False)

    predictors = [
        "netmhcpan4.ba",
        "netmhcpan4.el",
        "mixmhcpred",
        "mhcflurry_production_affinity",
        "presentation_without_flanks_presentation_score",
        "presentation_with_flanks_presentation_score",
        "presentation_without_flanks_processing_score",
        "presentation_with_flanks_processing_score",
    ]
    rows = []
    for sample_index, sample_id in enumerate(["sample1", "sample2", "sample3"]):
        for length_label, length in [("All", numpy.nan), ("8-mer", 8), ("9-mer", 9)]:
            for predictor_index, predictor in enumerate(predictors):
                base = 0.55 + sample_index * 0.02 + predictor_index * 0.01
                if length_label != "All":
                    base -= 0.03
                rows.append({
                    "sample_id": sample_id,
                    "length": length,
                    "length_label": length_label,
                    "predictor": predictor,
                    "ppv": min(0.95, base),
                    "auc": min(0.99, base + 0.15),
                    "percent_change_auc_ba": predictor_index + 1.0,
                    "percent_change_ppv_ba": predictor_index + 2.0,
                    "percent_change_auc_el": predictor_index + 3.0,
                    "percent_change_ppv_el": predictor_index + 4.0,
                    "percent_change_auc_mixmhcpred": predictor_index + 5.0,
                    "percent_change_ppv_mixmhcpred": predictor_index + 6.0,
                })
    pandas.DataFrame(rows).to_csv(
        artifacts / "accuracy_scores.multiallelic.csv", index=False)

    args = paper_figures.make_parser().parse_args([
        "--scores-dir", str(artifacts),
        "--out", str(tmp_path / "paper"),
        "--formats", "svg,pdf,png",
    ])
    assert paper_figures.run(args) == 0

    for path in [
        tmp_path / "paper" / "svg" /
        "fig.3_scores_plots_multiallelic.scatter.auc.ba.svg",
        tmp_path / "paper" / "pdf" /
        "fig.3_scores_plots_multiallelic.scatter.ppv.presentation.pdf",
        tmp_path / "paper" / "png" /
        "fig.3_scores_plots_multiallelic.bar.auc.presentation.png",
        tmp_path / "paper" / "paper_figures.pdf",
        tmp_path / "paper" / "manifest.csv",
        tmp_path / "paper" / "missing_inputs.md",
    ]:
        assert path.is_file()
    manifest = pandas.read_csv(tmp_path / "paper" / "manifest.csv")
    assert (
        manifest["figure"]
        == "fig.3_scores_plots_multiallelic.scatter.auc.ba"
    ).any()
    assert (
        manifest["figure"]
        == "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.ba"
    ).any()
    assert "skipped" in set(manifest["status"])
    assert "failed" not in set(manifest["status"])


def test_paper_figures_derives_scores_from_saved_predictions(tmp_path):
    pytest.importorskip("matplotlib")

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    rows = []
    peptides = {
        8: ("AAAAAAAK", "AAAAAAAL", "AAAAAAAM", "AAAAAAAN"),
        9: ("AAAAAAAAK", "AAAAAAAAL", "AAAAAAAAM", "AAAAAAAAN"),
    }
    for sample_id, sample_offset in (("s1", 0), ("s2", 5)):
        for length, peptide_values in peptides.items():
            for row_index, peptide in enumerate(peptide_values):
                hit = 1 if row_index < 2 else 0
                good = 0.90 - sample_offset * 0.01 - row_index * 0.02
                bad = 0.20 + row_index * 0.02
                rows.append({
                    "sample_id": sample_id,
                    "sample_group": "MULTIALLELIC-RECENT",
                    "peptide": peptide,
                    "length": length,
                    "hit": hit,
                    "netmhcpan4.ba": 20.0 if hit else 2000.0,
                    "netmhcpan4.el": 0.10 if hit else 10.0,
                    "mixmhcpred": good - 0.05 if hit else bad + 0.05,
                    "mhcflurry_production": 30.0 if hit else 3000.0,
                    "presentation_without_flanks_presentation_score": (
                        good if hit else bad),
                    "presentation_with_flanks_presentation_score": (
                        good + 0.01 if hit else bad),
                    "presentation_without_flanks_processing_score": (
                        good - 0.03 if hit else bad),
                    "presentation_with_flanks_processing_score": (
                        good - 0.02 if hit else bad),
                })
    predictions = scores_dir / "benchmark.multiallelic.csv"
    pandas.DataFrame(rows).to_csv(predictions, index=False)

    args = paper_figures.make_parser().parse_args([
        "--scores-dir", str(scores_dir),
        "--multiallelic-predictions", str(predictions),
        "--out", str(tmp_path / "paper"),
        "--formats", "png",
        "--combined-pdf", "none",
    ])
    assert paper_figures.run(args) == 0
    manifest = pandas.read_csv(tmp_path / "paper" / "manifest.csv")
    assert (
        manifest["figure"]
        == "fig.3_scores_plots_multiallelic.scatter.auc.ba"
    ).any()
    assert (
        tmp_path / "paper" / "png" /
        "fig.3_scores_plots_multiallelic.scatter.auc.ba.png"
    ).is_file()


def test_paper_figures_sample_groups_use_explicit_predictions(tmp_path):
    class CaptureWriter:
        def __init__(self):
            self.rows = []

        def fail(self, family, figure, note):
            self.rows.append({
                "family": family,
                "figure": figure,
                "status": "failed",
                "note": note,
            })

        def skip(self, family, figure, missing, note):
            self.rows.append({
                "family": family,
                "figure": figure,
                "status": "skipped",
                "missing": missing,
                "note": note,
            })

    scores_dir = tmp_path / "scores"
    predictions_dir = tmp_path / "predictions"
    scores_dir.mkdir()
    predictions_dir.mkdir()
    predictions = predictions_dir / "benchmark.multiallelic.csv"
    pandas.DataFrame([
        {
            "sample_id": "recent",
            "sample_group": "MULTIALLELIC-RECENT",
            "peptide": "SIINFEKL",
            "hit": 1,
            "mhcflurry_production": 0.9,
        },
        {
            "sample_id": "old",
            "sample_group": "MULTIALLELIC-OLD",
            "peptide": "SLYNTVATL",
            "hit": 0,
            "mhcflurry_production": 0.1,
        },
    ]).to_csv(predictions, index=False)
    writer = CaptureWriter()
    args = paper_figures.make_parser().parse_args([
        "--scores-dir", str(scores_dir),
        "--multiallelic-predictions", str(predictions),
        "--out", str(tmp_path / "paper"),
    ])

    inputs = paper_figures._resolve_figure_inputs(args, writer)
    sample_ids = paper_figures._read_sample_group_ids(args, inputs, writer)

    assert sample_ids == {"recent"}
    assert writer.rows == []


def test_paper_figures_uses_current_comparison_when_scores_absent(tmp_path):
    pytest.importorskip("matplotlib")

    comparison = tmp_path / "comparison"
    (comparison / "affinity").mkdir(parents=True)
    (comparison / "side_a.json").write_text(json.dumps({"label": "current"}))
    (comparison / "side_b.json").write_text(json.dumps({"label": "public"}))
    pandas.DataFrame([
        {
            "allele": "HLA-A*02:01",
            "n": 40,
            "n_pos": 10,
            "a_roc_auc": 0.95,
            "b_roc_auc": 0.90,
            "a_ppv_at_n": 0.80,
            "b_ppv_at_n": 0.70,
        },
        {
            "allele": "HLA-B*07:02",
            "n": 40,
            "n_pos": 10,
            "a_roc_auc": 0.91,
            "b_roc_auc": 0.88,
            "a_ppv_at_n": 0.76,
            "b_ppv_at_n": 0.72,
        },
    ]).to_csv(comparison / "affinity" / "per_allele.csv", index=False)

    args = paper_figures.make_parser().parse_args([
        "--comparison-dir", str(comparison),
        "--out", str(tmp_path / "paper"),
        "--formats", "png",
        "--combined-pdf", "none",
    ])
    assert paper_figures.run(args) == 0
    manifest = pandas.read_csv(tmp_path / "paper" / "manifest.csv")
    assert (
        manifest["figure"]
        == "fig.1_model_selection_predictor_accuracy.scores.hla_a"
    ).any()
    assert (
        manifest["figure"]
        == "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.ba"
    ).any()


def test_current_comparison_labels_counts_as_evaluation_peptides(
        tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    comparison = tmp_path / "comparison"
    (comparison / "affinity").mkdir(parents=True)
    pandas.DataFrame([{
        "allele": "HLA-A*02:01",
        "n": 40,
        "n_pos": 10,
        "a_roc_auc": 0.95,
        "b_roc_auc": 0.90,
        "a_ppv_at_n": 0.80,
        "b_ppv_at_n": 0.70,
    }]).to_csv(comparison / "affinity" / "per_allele.csv", index=False)

    class CaptureWriter:
        def __init__(self):
            self.xlabels = {}

        def save(self, fig, name, _family, note=""):
            self.xlabels[name] = [ax.get_xlabel() for ax in fig.axes]
            plt.close(fig)

        def skip(self, *_args, **_kwargs):
            pass

    writer = CaptureWriter()
    inputs = paper_figures.FigureInputs(
        scores_dir=tmp_path / "scores",
        comparison_dir=comparison,
        run_dir=None,
        multiallelic_predictions=None,
        monoallelic_predictions=None,
    )
    paper_figures._generate_model_selection_figures(inputs, writer)

    xlabels = writer.xlabels[
        "fig.1_model_selection_predictor_accuracy.scores.hla_a"]
    assert "Evaluation peptides" in xlabels
    assert "Training peptides" not in xlabels


def test_current_model_information_uses_only_final_manifests(tmp_path):
    run_dir = tmp_path / "run"
    final_paths = [
        run_dir / "affinity" / "models.combined" / "manifest.csv",
        run_dir / "processing" / "models.selected.with_flanks" /
        "manifest.csv",
        run_dir / "processing" / "models.selected.no_flank" /
        "manifest.csv",
    ]
    stale_paths = [
        run_dir / "affinity" / "models.unselected.combined" / "manifest.csv",
        run_dir / "processing" / "models.unselected.with_flanks" /
        "manifest.csv",
        run_dir / "presentation" / "models" / "affinity_predictor" /
        "manifest.csv",
        run_dir / "scratch" / "manifest.csv",
    ]
    for path in final_paths + stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model_name\nmodel_1\n")
    presentation_weights = run_dir / "presentation" / "models" / "weights.csv"
    presentation_weights.parent.mkdir(parents=True, exist_ok=True)
    presentation_weights.write_text(
        ",intercept\nwith_flanks,1.0\nwithout_flanks,2.0\n")

    assert set(paper_figures._final_model_manifest_paths(run_dir)) == set(
        final_paths)
    counts = paper_figures._current_model_counts(run_dir).set_index(
        "component")["models"].to_dict()
    assert counts == {
        "Affinity": 1,
        "Presentation": 2,
        "Processing no flank": 1,
        "Processing with flanks": 1,
    }


def test_paper_figures_rerender_clears_command_owned_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    out = tmp_path / "paper"
    for path in [
            out / "svg" / "old.svg",
            out / "pdf" / "old.pdf",
            out / "png" / "old.png",
            out / "assets" / "old.svg"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")
    for path in [
            out / "paper_figures.pdf",
            out / "old_custom_combined.pdf",
            out / "manifest.csv",
            out / "missing_inputs.md"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")
    unrelated = out / "README.keep"
    unrelated.write_text("user file")

    args = paper_figures.make_parser().parse_args([
        "--artifacts-dir", str(artifacts),
        "--out", str(out),
        "--formats", "png",
        "--combined-pdf", "none",
    ])
    assert paper_figures.run(args) == 0

    assert not (out / "svg").exists()
    assert not (out / "pdf").exists()
    assert not (out / "png" / "old.png").exists()
    assert not (out / "assets").exists()
    assert not (out / "paper_figures.pdf").exists()
    assert not (out / "old_custom_combined.pdf").exists()
    assert pandas.read_csv(out / "manifest.csv").shape[0] > 0
    assert unrelated.read_text() == "user file"


def test_paper_figures_rejects_scores_output_overlap_without_cleanup(tmp_path):
    out = tmp_path / "paper"
    source_asset = out / "assets" / "source.svg"
    source_pdf = out / "source.pdf"
    source_asset.parent.mkdir(parents=True)
    source_asset.write_bytes(b"source asset")
    source_pdf.write_bytes(b"source pdf")
    args = paper_figures.make_parser().parse_args([
        "--scores-dir", str(out),
        "--out", str(out),
    ])

    with pytest.raises(SystemExit, match="would delete input path"):
        paper_figures.run(args)

    assert source_asset.read_bytes() == b"source asset"
    assert source_pdf.read_bytes() == b"source pdf"
    assert not (out / "manifest.csv").exists()


def test_paper_figures_rejects_prediction_in_cleanup_tree(tmp_path):
    scores = tmp_path / "scores"
    scores.mkdir()
    out = tmp_path / "paper"
    prediction = out / "assets" / "predictions.csv"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("sample_id,hit,score\ns1,1,0.9\n")
    args = paper_figures.make_parser().parse_args([
        "--scores-dir", str(scores),
        "--multiallelic-predictions", str(prediction),
        "--out", str(out),
    ])

    with pytest.raises(SystemExit, match="would delete input path"):
        paper_figures.run(args)

    assert prediction.is_file()
    assert not (out / "manifest.csv").exists()


def test_paper_figures_monoallelic_scatter_uses_all_length_rows(monkeypatch):
    captured = {}

    def fake_scatter(pivot, *_args, **_kwargs):
        captured["pivot"] = pivot

    monkeypatch.setattr(
        paper_figures, "_plot_scatter_triptych_from_pivot", fake_scatter)

    scores = pandas.DataFrame([
        {
            "allele": "HLA-A*02:01",
            "length": numpy.nan,
            "length_label": "All",
            "predictor": "candidate",
            "auc": 0.90,
        },
        {
            "allele": "HLA-A*02:01",
            "length": numpy.nan,
            "length_label": "All",
            "predictor": "baseline",
            "auc": 0.80,
        },
        {
            "allele": "HLA-A*02:01",
            "length": 8,
            "length_label": "8-mer",
            "predictor": "candidate",
            "auc": 0.10,
        },
        {
            "allele": "HLA-A*02:01",
            "length": 8,
            "length_label": "8-mer",
            "predictor": "baseline",
            "auc": 0.20,
        },
    ])
    predictors = paper_figures.PredictorConfig(
        candidate="candidate",
        external_baselines=(("baseline", "baseline"),),
        preferred_predictors=("candidate", "baseline"),
        presentation_panel_predictors=("candidate",),
        presentation_panel_baselines=("baseline",),
    )

    class Writer:
        def skip(self, *args):
            raise AssertionError("unexpected skip: %r" % (args,))

    paper_figures._plot_monoallelic_scatter(
        scores,
        pandas.DataFrame(),
        "auc",
        "AUC",
        100,
        "mono",
        Writer(),
        predictors,
        preferred_candidate="candidate",
    )
    pivot = captured["pivot"]
    assert pivot.loc["HLA-A*02:01", "candidate"] == 0.90
    assert pivot.loc["HLA-A*02:01", "baseline"] == 0.80
    assert len(pivot) == 1


def test_paper_figures_prediction_scoring_drops_invalid_hit_rows():
    group = pandas.DataFrame({
        "sample_id": ["s1", "s1", "s1", "s1"],
        "peptide": ["AAAA", "BBBB", "CCCC", "DDDD"],
        "hit": [1, 0, numpy.nan, "bad"],
        "candidate": [0.9, 0.1, 1.0, 1.0],
    })
    rows = paper_figures._scores_for_prediction_group(
        group,
        index_column="sample_id",
        group_value="s1",
        length=None,
        length_label="All",
        predictor_columns=["candidate"],
        predictor_orientations={"candidate": True},
    )
    assert rows[0]["auc"] == 1.0
    assert rows[0]["ppv"] == 1.0


def test_paper_figures_prediction_scoring_uses_shared_finite_rows():
    group = pandas.DataFrame({
        "sample_id": ["s1"] * 4,
        "peptide": ["AAAA", "BBBB", "CCCC", "DDDD"],
        "hit": [1, 1, 0, 0],
        # On all rows candidate has AUC=.75 and PPV=.5. The row baseline
        # cannot score is a difficult positive; both predictors must use the
        # resulting shared three-row subset, where both metrics are 1.
        "candidate": [0.9, 0.1, 0.8, 0.0],
        "baseline": [0.9, numpy.nan, 0.2, 0.1],
    })
    rows = paper_figures._scores_for_prediction_group(
        group,
        index_column="sample_id",
        group_value="s1",
        length=None,
        length_label="All",
        predictor_columns=["candidate", "baseline"],
        predictor_orientations={"candidate": True, "baseline": True},
    )

    assert {row["predictor"] for row in rows} == {"candidate", "baseline"}
    assert {row["n"] for row in rows} == {3}
    assert {row["n_pos"] for row in rows} == {1}
    assert {row["auc"] for row in rows} == {1.0}
    assert {row["ppv"] for row in rows} == {1.0}


def test_paper_figures_monoallelic_scoring_prefers_allele(tmp_path):
    predictions = tmp_path / "predictions.csv"
    pandas.DataFrame([
        {
            "sample_id": "shared",
            "allele": "HLA-A*02:01",
            "peptide": "AAAAAAAAA",
            "hit": 1,
            "candidate": 0.9,
        },
        {
            "sample_id": "shared",
            "allele": "HLA-A*02:01",
            "peptide": "AAAAAAAAB",
            "hit": 0,
            "candidate": 0.1,
        },
        {
            "sample_id": "shared",
            "allele": "HLA-B*07:02",
            "peptide": "BBBBBBBBB",
            "hit": 1,
            "candidate": 0.8,
        },
        {
            "sample_id": "shared",
            "allele": "HLA-B*07:02",
            "peptide": "BBBBBBBBA",
            "hit": 0,
            "candidate": 0.2,
        },
    ]).to_csv(predictions, index=False)

    predictor_info = pandas.DataFrame([{
        "predictor": "candidate",
        "higher_is_better": True,
    }]).set_index("predictor", drop=False)
    scores = paper_figures.score_saved_prediction_table(
        predictions, kind="monoallelic", predictor_info=predictor_info)
    all_scores = scores.loc[
        (scores["length_label"] == "All") &
        (scores["predictor"] == "candidate")
    ]

    assert set(all_scores["allele"]) == {"HLA-A*02:01", "HLA-B*07:02"}
    assert set(all_scores["auc"]) == {1.0}
    assert len(all_scores) == 2


def test_paper_figures_ppv_uses_tie_breaker():
    # File-order-stable sorting would pick the two hit rows first. The explicit
    # tie breaker randomizes equal-score ordering deterministically.
    ppv = paper_figures._ppv_at_n(
        numpy.array([1, 1, 0, 0]),
        numpy.array([0.5, 0.5, 0.5, 0.5]),
        2,
        tie_breaker=numpy.array([0.8, 0.9, 0.1, 0.2]),
    )
    assert ppv == 0.0


def test_paper_figures_resolves_run_dir_for_named_comparison_dir(tmp_path):
    run_dir = tmp_path / "run"
    comparison_dir = run_dir / "comparison"
    comparison_dir.mkdir(parents=True)
    (run_dir / "processing").mkdir()
    assert paper_figures._resolve_run_dir(comparison_dir) == run_dir


def test_paper_figures_external_baseline_geometry_is_configurable(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    predictor_info = pandas.DataFrame([
        {
            "predictor": predictor,
            "short": predictor.replace("_", " "),
            "description": predictor,
            "color": "(0.5, 0.4, 0.8)",
        }
        for predictor in ("candidate", "baseline_a", "baseline_b", "baseline_c")
    ]).set_index("predictor")
    scores = pandas.DataFrame([
        {
            "sample_id": sample_id,
            "length_label": "All",
            "predictor": predictor,
            "auc": value,
            "ppv": value - 0.1,
            "percent_change_auc_a": 1.0,
            "percent_change_auc_b": 2.0,
        }
        for sample_id, offset in (("s1", 0.0), ("s2", 0.02))
        for predictor, value in (
            ("candidate", 0.75 + offset),
            ("baseline_a", 0.70 + offset),
            ("baseline_b", 0.72 + offset),
        )
    ])
    predictors = paper_figures.PredictorConfig(
        candidate="candidate",
        external_baselines=(
            ("baseline_a", "a"),
            ("baseline_b", "b"),
            ("baseline_c", "c"),
        ),
        preferred_predictors=(
            "candidate", "baseline_a", "baseline_b", "baseline_c"),
        presentation_panel_predictors=("candidate",),
        presentation_panel_baselines=("baseline_a", "baseline_b", "baseline_c"),
    )
    saved_axes = {}

    class CaptureWriter:
        def save(self, fig, name, _family, note=""):
            saved_axes[name] = len(fig.axes)
            plt.close(fig)

        def skip(self, _family, name, _missing, _note):
            saved_axes[name] = "skipped"

    writer = CaptureWriter()
    paper_figures._apply_paper_style()
    paper_figures._plot_external_scatter_triptych(
        scores, predictor_info, "auc", "AUC", "external", 100, writer,
        predictors)
    paper_figures._plot_percent_change_by_length(
        scores, predictor_info, "auc", "AUC", "by_length", writer,
        predictors)
    paper_figures._plot_percent_change_bars(
        scores, predictor_info, "auc", "AUC", "bars", writer, predictors)
    paper_figures._plot_scatter_triptych_from_pivot(
        paper_figures._pivot_all_lengths(scores, "auc"),
        predictor_info,
        "candidate",
        "AUC",
        100,
        "pivot",
        "test",
        writer,
        predictors,
    )
    assert saved_axes == {
        "external": 2,
        "by_length": 2,
        "bars": 2,
        "pivot": 2,
    }


def test_paper_figures_render_failure_returns_nonzero(
        tmp_path, monkeypatch):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(
        paper_figures, "_generate_multiallelic_figures", fail_render)
    args = paper_figures.make_parser().parse_args([
        "--scores-dir", str(scores_dir),
        "--out", str(tmp_path / "paper"),
        "--formats", "png",
        "--combined-pdf", "none",
    ])

    assert paper_figures.run(args) == 2
    manifest = pandas.read_csv(tmp_path / "paper" / "manifest.csv")
    failed = manifest.loc[manifest.status == "failed"]
    assert len(failed) == 1
    assert failed.iloc[0]["family"] == "multiallelic"
    assert "render exploded" in failed.iloc[0]["note"]


def test_paper_figures_bad_predictor_config_writes_manifest(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    args = paper_figures.make_parser().parse_args([
        "--artifacts-dir", str(artifacts),
        "--out", str(tmp_path / "paper"),
        "--external-baselines", "",
    ])
    assert paper_figures.run(args) == 2
    manifest = pandas.read_csv(tmp_path / "paper" / "manifest.csv")
    assert manifest.iloc[0]["family"] == "configuration"
    assert manifest.iloc[0]["figure"] == "predictor_config"
    assert manifest.iloc[0]["status"] == "failed"


def test_paper_figures_bad_predictor_config_preserves_existing_suite(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    out_dir = tmp_path / "paper"
    out_dir.mkdir()
    combined = out_dir / "paper_figures.pdf"
    manifest = out_dir / "manifest.csv"
    combined.write_bytes(b"existing suite")
    manifest.write_text("existing manifest\n")
    args = paper_figures.make_parser().parse_args([
        "--artifacts-dir", str(artifacts),
        "--out", str(out_dir),
        "--external-baselines", "",
    ])

    assert paper_figures.run(args) == 2
    assert combined.read_bytes() == b"existing suite"
    assert manifest.read_text() == "existing manifest\n"


@pytest.mark.parametrize("formats", ["pdf,pdf", "pdf,"])
def test_paper_figures_rejects_duplicate_or_empty_formats(formats):
    with pytest.raises(ValueError):
        paper_figures._parse_formats(formats)


@pytest.mark.parametrize("reserved_name", [
    "manifest.csv", "missing_inputs.md", "pdf", "assets", "pdf/figure.pdf",
])
def test_paper_figures_rejects_combined_pdf_metadata_collisions(
        tmp_path, reserved_name):
    with pytest.raises(SystemExit, match="collides"):
        paper_figures._validate_paper_output_paths(
            tmp_path, tmp_path / reserved_name)


def test_paper_figures_predictor_config_parser():
    args = paper_figures.make_parser().parse_args([
        "--artifacts-dir", "artifacts",
        "--out", "out",
        "--candidate-predictor", "candidate",
        "--external-baselines", "baseline_a:ba,baseline_b",
        "--preferred-predictors", "candidate,baseline_a",
        "--presentation-panel-predictors", "candidate_ps",
        "--presentation-panel-baselines", "baseline_a,baseline_b",
    ])
    config = paper_figures._parse_predictor_config(args)
    assert config.candidate == "candidate"
    assert config.external_baselines == (
        ("baseline_a", "ba"),
        ("baseline_b", "baseline_b"),
    )
    assert config.preferred_predictors == ("candidate", "baseline_a")
    assert config.presentation_panel_predictors == ("candidate_ps",)
    assert config.presentation_panel_baselines == ("baseline_a", "baseline_b")
