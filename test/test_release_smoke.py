"""Release smoke mode shrinks scale without changing stages, and only when asked."""

import importlib.util
from pathlib import Path
import re
import subprocess

import pandas
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = [REPO / "scripts/training/pan_allele_release_full.sh",
           REPO / "scripts/training/pan_allele_release_affinity.sh"]
GUARD = 'if [ "${MHCFLURRY_RELEASE_SMOKE:-0}" = "1" ]; then'


@pytest.fixture
def smoke():
    spec = importlib.util.spec_from_file_location("release_smoke", REPO / "scripts/training/release_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cap_hyperparameters_truncates_and_caps_every_epoch_count(smoke):
    affinity = {"max_epochs": 5000, "patience": 20, "train_data": {
        "pretrain": True, "pretrain_max_epochs": 50, "pretrain_min_epochs": 5,
        "pretrain_steps_per_epoch": 256, "pretrain_peptides_per_epoch": 64}}
    processing = {"convolutional_kernel_size": 13}  # no max_epochs key at all
    capped = smoke.cap_hyperparameters([affinity, processing, dict(affinity)], 2, 2)
    assert len(capped) == 2
    assert capped[0]["max_epochs"] == 2 and capped[0]["patience"] == 20
    assert capped[0]["train_data"] == {
        "pretrain": True, "pretrain_max_epochs": 2, "pretrain_min_epochs": 2,
        "pretrain_steps_per_epoch": 16, "pretrain_peptides_per_epoch": 64}
    assert capped[1] == {"convolutional_kernel_size": 13, "max_epochs": 2}
    assert affinity["max_epochs"] == 5000  # inputs are not mutated
    with pytest.raises(ValueError):
        smoke.cap_hyperparameters([], 2, 2)


def test_cap_hyperparameters_cli_rewrites_file_in_place(smoke, tmp_path):
    path = tmp_path / "hyperparameters.yaml"
    path.write_text(yaml.safe_dump([{"max_epochs": 500}] * 5))
    assert smoke.main(["cap-hyperparameters", str(path), "--max-architectures", "2", "--max-epochs", "2"]) == 0
    assert yaml.safe_load(path.read_text()) == [{"max_epochs": 2}, {"max_epochs": 2}]


def test_sample_exclusions_keep_a_seeded_subset_and_always_exclude_holdout(smoke, tmp_path):
    hits = ["big%d" % index for index in range(30) for _ in range(250)] + ["tiny"] * 10 + ["held"] * 500
    excluded, kept = smoke.smoke_sample_exclusions(hits, ["held"], keep=20, seed=42)
    assert len(kept) == 20 and "held" in excluded and "held" not in kept and "tiny" not in kept
    assert set(excluded) | set(kept) == set(hits) and not set(excluded) & set(kept)
    assert smoke.smoke_sample_exclusions(hits, ["held"], keep=20, seed=42) == (excluded, kept)
    with pytest.raises(ValueError, match="need 31"):
        smoke.smoke_sample_exclusions(hits, ["held"], keep=31, seed=42)
    pandas.DataFrame({"sample_id": hits}).to_csv(tmp_path / "hits.csv", index=False)
    pandas.DataFrame({"sample_id": ["held"]}).to_csv(tmp_path / "holdout.csv", index=False)
    out = tmp_path / "excluded.csv"
    assert smoke.main(["sample-exclusions", "--hits", str(tmp_path / "hits.csv"), "--holdout",
                       str(tmp_path / "holdout.csv"), "--keep", "20", "--seed", "42", "--out", str(out)]) == 0
    assert sorted(pandas.read_csv(out).sample_id) == excluded


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_release_scripts_parse_and_only_call_the_helper_in_smoke_mode(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    lines = script.read_text().splitlines()
    calls = [index for index, line in enumerate(lines) if '"$SMOKE_HELPER"' in line]
    assert calls, "smoke hooks missing from %s" % script.name
    for index in calls:
        previous = next(line.strip() for line in reversed(lines[:index]) if line.strip())
        assert previous == GUARD, (script.name, index + 1, lines[index])


def test_unset_smoke_defaults_reproduce_the_previous_literals():
    full = SCRIPTS[0].read_text()
    assert 'PRESENTATION_CALIBRATION_PEPTIDES_PER_LENGTH="${PRESENTATION_CALIBRATION_PEPTIDES_PER_LENGTH:-10000}"' in full
    assert 'PRESENTATION_CALIBRATION_GENOTYPES="${PRESENTATION_CALIBRATION_GENOTYPES:-50}"' in full
    assert 'PROCESSING_EXCLUDE_SAMPLES="$RELEASE_HOLDOUT_DIR/processing_samples.csv"' in full
    affinity = SCRIPTS[1].read_text()
    assert re.search(r'^AFFINITY_EVAL_LIMIT_ARGS=\(\)$', affinity, re.M)
    # Every smoke-only assignment sits inside the guard.
    for text in (full, affinity):
        body = text.split(GUARD)
        assert len(body) > 1


def test_processing_data_is_reused_on_relaunch_only_after_validation():
    full = SCRIPTS[0].read_text()
    guard = full.index('mhcflurry train validate-processing-data --data "$(pwd)/train_data.csv.bz2"; then')
    generate = full.index("mhcflurry train processing-data \\")
    assert guard < generate
    between = full[guard:generate]
    assert "Reusing validated processing training data" in between and "\nelse\n" in between


def bash_block(text, start, end="fi\n"):
    """One top-level shell block from the release script, verbatim."""
    begin = text.index(start)
    stop = text.index("\n" + end, begin) + 1 + len(end)
    return text[begin:stop]


def run_block(block, cwd, prelude=""):
    script = "set -euo pipefail\n" + prelude + block
    return subprocess.run(["bash", "-c", script], cwd=cwd, capture_output=True,
                          text=True, check=True).stdout


def test_annotated_hits_are_reused_so_relaunch_keeps_the_recorded_hash(tmp_path):
    block = bash_block(SCRIPTS[0].read_text(), 'if [ -s "$(pwd)/hits_with_tpm.csv.bz2" ]')
    prelude = ('python() { echo "ANNOTATED"; }\n'
               'compress_csv_bzip2() { echo "COMPRESSED"; }\n'
               'mhcflurry-downloads() { echo /stub; }\n')
    assert "ANNOTATED" in run_block(block, tmp_path, prelude)
    subprocess.run(["bzip2", "-c"], input=b"sample_id,peptide\nS,SIINFEKL\n",
                   stdout=(tmp_path / "hits_with_tpm.csv.bz2").open("wb"), check=True)
    reused = run_block(block, tmp_path, prelude)
    assert "Reusing annotated hits" in reused and "ANNOTATED" not in reused
    (tmp_path / "hits_with_tpm.csv.bz2").write_bytes(b"truncated garbage")
    assert "ANNOTATED" in run_block(block, tmp_path, prelude)


def test_matched_preparation_resumes_from_saved_pools_and_clears_empty_artifacts(tmp_path):
    block = bash_block(SCRIPTS[0].read_text(), "PROCESSING_DATA_RESUME_ARGS=()", "    fi\n")
    report = 'echo "args=${PROCESSING_DATA_RESUME_ARGS[*]:-none}"\n'
    assert "args=none" in run_block(block + report, tmp_path)
    matching = tmp_path / "train_data.csv.matching"
    matching.mkdir()
    assert "args=none" in run_block(block + report, tmp_path)
    assert not matching.exists()  # nothing scored yet, so nothing is preserved
    matching.mkdir()
    (matching / "experiment.json").write_text("{}")
    (matching / "abc.candidate_pool.csv.bz2").write_bytes(b"pool")
    assert "args=--resume" in run_block(block + report, tmp_path)
    assert (matching / "abc.candidate_pool.csv.bz2").is_file()


@pytest.mark.parametrize("script", SCRIPTS)
def test_optional_argument_arrays_expand_when_empty_under_nounset(script):
    """`set -u` with an empty array aborts on bash 3.2 without the +alternate form."""
    text = script.read_text()
    for name in re.findall(r"^\s*([A-Z_]+)=\(\)\s*$", text, flags=re.MULTILINE):
        safe = '${%s[@]+"${%s[@]}"}' % (name, name)
        assert text.count('"${%s[@]}"' % name) == text.count(safe), name
        if safe in text:
            run_block('%s=()\nprintf "arg=%%s\\n" %s\n' % (name, safe), script.parent)
