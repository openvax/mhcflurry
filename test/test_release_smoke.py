"""Release smoke mode shrinks scale without changing stages, and only when asked."""

import importlib.util
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

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
    # The CLI reads the eligibility columns too, so write a full hit table.
    pandas.DataFrame({
        "sample_id": hits, "mhc_class": "I", "peptide": "SIINFEKL",
        "protein_ensembl": "ENSP1", "format": "MONOALLELIC",
        "hla": "HLA-A*02:01"}).to_csv(tmp_path / "hits.csv", index=False)
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
    prelude = ('RELEASE_RANDOM_SEED=42\n'
               'python() { case "$*" in *--validate-existing*) echo "VALIDATED";; '
               '*) echo "ANNOTATED";; esac; }\n'
               'compress_csv_bzip2() { echo "COMPRESSED"; }\n'
               'mhcflurry-downloads() { echo /stub; }\n')
    assert "ANNOTATED" in run_block(block, tmp_path, prelude)
    subprocess.run(["bzip2", "-c"], input=b"sample_id,peptide\nS,SIINFEKL\n",
                   stdout=(tmp_path / "hits_with_tpm.csv.bz2").open("wb"), check=True)
    reused = run_block(block, tmp_path, prelude)
    assert "Reusing annotated hits" in reused and "ANNOTATED" not in reused
    assert "VALIDATED" in reused
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


def test_data_vintage_selects_a_download_label_and_rejects_other_values(tmp_path, monkeypatch):
    """Only the release pipeline's two vintages are accepted, and only one exports."""
    monkeypatch.delenv("MHCFLURRY_DOWNLOADS_CURRENT_RELEASE", raising=False)
    block = bash_block(SCRIPTS[0].read_text(),
                       'MHCFLURRY_RELEASE_DATA_VINTAGE="${MHCFLURRY_RELEASE_DATA_VINTAGE:-current}"',
                       "esac\n")
    report = 'echo "label=${MHCFLURRY_DOWNLOADS_CURRENT_RELEASE:-unset}"\n'
    assert "label=unset" in run_block(block + report, tmp_path)
    assert "label=unset" in run_block(
        block + report, tmp_path, 'MHCFLURRY_RELEASE_DATA_VINTAGE=current\n')
    assert "label=custom" in run_block(
        block + report, tmp_path, 'MHCFLURRY_DOWNLOADS_CURRENT_RELEASE=custom\n')
    assert "label=2.0.0" in run_block(
        block + report, tmp_path, 'MHCFLURRY_RELEASE_DATA_VINTAGE=public-2020\n')
    rejected = subprocess.run(
        ["bash", "-c", "set -euo pipefail\nMHCFLURRY_RELEASE_DATA_VINTAGE=2019\n" + block],
        cwd=tmp_path, capture_output=True, text=True)
    assert rejected.returncode == 2 and "must be current or public-2020" in rejected.stderr


def test_data_vintage_is_recorded_before_any_training():
    """The run keeps the resolved training inputs, hashed, next to its config."""
    full = SCRIPTS[0].read_text()
    record = full.index("config/data_vintage.json")
    for later in ("=== STAGE 1: AFFINITY", "release-holdout build"):
        assert record < full.index(later), later
    assert "curated_training_data" in full and "annotated_ms" in full


def test_fresh_cache_vintage_records_actual_allele_csv(tmp_path):
    full = SCRIPTS[0].read_text()
    start = full.index("mhcflurry-downloads fetch data_evaluation data_curated data_mass_spec_annotated")
    stop = full.index("\nPYVINTAGE", start) + len("\nPYVINTAGE\n")
    (tmp_path / "curated_training_data.csv.bz2").write_bytes(b"curated")
    (tmp_path / "annotated_ms.csv.bz2").write_bytes(b"mass spec")
    prelude = '''BASE_OUT="$PWD"
MHCFLURRY_RELEASE_DATA_VINTAGE=current
mhcflurry-downloads() {
    if [ "$1" = fetch ]; then
        for item in "$@"; do
            if [ "$item" = allele_sequences ]; then
                echo 'allele,sequence' > "$PWD/allele.csv"
            fi
        done
    else
        if [ "$2" = allele_sequences ]; then test -s "$PWD/allele.csv" || return 1; fi
        echo "$PWD"
    fi
}
mhcflurry() { echo "$PWD/allele.csv"; }
'''
    prelude += "python() { " + shlex.quote(sys.executable) + ' "$@"; }\n'
    run_block(full[start:stop], tmp_path, prelude)
    record = json.loads((tmp_path / "config/data_vintage.json").read_text())
    assert record["allele_sequences_dir"] == str(tmp_path)
    assert record["allele_sequences"] == {
        "path": str(tmp_path / "allele.csv"),
        "sha256": hashlib.sha256(b"allele,sequence\n").hexdigest()}


def ineligible_and_eligible_hits():
    """One eligible sample plus one row of every reason the real command drops."""
    rows = [
        # sample_id, mhc_class, peptide, protein_ensembl, format, hla
        ("keep", "I", "SIINFEKL", "ENSP1", "MONOALLELIC", "HLA-A*02:01"),
        ("class_ii", "II", "SIINFEKL", "ENSP1", "MONOALLELIC", "HLA-A*02:01"),
        ("too_long", "I", "SIINFEKLSIIN", "ENSP1", "MONOALLELIC", "HLA-A*02:01"),
        ("no_protein", "I", "SIINFEKL", None, "MONOALLELIC", "HLA-A*02:01"),
        ("odd_residue", "I", "SIINFEKX", "ENSP1", "MONOALLELIC", "HLA-A*02:01"),
        ("multiallelic", "I", "SIINFEKL", "ENSP1", "MULTIALLELIC", "HLA-A*02:01"),
        ("not_abc", "I", "SIINFEKL", "ENSP1", "MONOALLELIC", "HLA-E*01:01"),
        ("serotype", "I", "SIINFEKL", "ENSP1", "MONOALLELIC", "A2"),
    ]
    return pandas.DataFrame(rows, columns=[
        "sample_id", "mhc_class", "peptide", "protein_ensembl", "format", "hla"])


def test_eligible_hits_mirror_the_processing_commands_own_filters(smoke):
    """Smoke samples must survive to training; picking from raw hits did not."""
    frame = ineligible_and_eligible_hits()
    canonicalize = smoke.load_canonicalize_allele()
    eligible = smoke.eligible_processing_hits(frame, canonicalize)
    assert sorted(eligible.sample_id.unique()) == ["keep"]
    # A serotype is fatal in the real command, so it must not be selectable.
    with pytest.raises(ValueError):
        canonicalize("A2")


def test_sample_exclusions_keep_only_samples_that_reach_training(smoke, tmp_path):
    frame = pandas.concat([ineligible_and_eligible_hits()] * 250, ignore_index=True)
    extra = frame.loc[frame.sample_id == "keep"].copy()
    for name in ("keep2", "keep3", "holdout_sample"):
        extra["sample_id"] = name
        frame = pandas.concat([frame, extra], ignore_index=True)
    eligible = smoke.eligible_processing_hits(frame, smoke.load_canonicalize_allele())
    excluded, kept = smoke.smoke_sample_exclusions(
        eligible.sample_id, ["holdout_sample"], keep=2, seed=42, min_hits=200)
    assert len(kept) == 2 and "holdout_sample" not in kept
    assert set(kept) <= {"keep", "keep2", "keep3"}
    assert "holdout_sample" in excluded and "class_ii" not in kept


def test_smoke_reduces_the_processing_held_out_sample_count(tmp_path):
    """Holding out ten of twenty samples is what broke the first smoke run."""
    block = bash_block(SCRIPTS[0].read_text(),
                       'if [ "${MHCFLURRY_RELEASE_SMOKE:-0}" = "1" ]; then\n    PRESENTATION_SAMPLE_FRACTION=0.01')
    report = ('echo "kept=$SMOKE_PROCESSING_SAMPLES held_out=$PROCESSING_HELD_OUT_SAMPLES"\n')
    out = run_block(block + report, tmp_path, 'MHCFLURRY_RELEASE_SMOKE=1\n')
    assert "kept=20 held_out=4" in out
