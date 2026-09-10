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

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas
import pytest

from mhcflurry.proteome_decoys import (
    infer_flanking_length,
    iter_protein_peptide_records,
    load_reference_sequences,
    make_peptide_frame_for_accessions,
    sample_peptide_frame_for_accessions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_benchmark_alternates_comparison_order():
    module = load_script(REPO_ROOT / "scripts/training/benchmark_processing_preparation.py")
    calls = []

    def first():
        calls.append("first")
        return "a"

    def second():
        calls.append("second")
        return "b"

    results, timings, orders = module.timed_pair(first, second, 2)
    assert results == ["a", "b"]
    assert calls == ["first", "second", "second", "first"]
    assert orders == [[0, 1], [1, 0]]
    for measured in timings:
        assert len(measured["seconds"]) == len(measured["cpu_seconds"]) == 2


def test_preparation_benchmark_saves_parity_timings_and_refuses_overwrite(tmp_path):
    import json
    from mhcflurry.cli import main as cli_main
    path = tmp_path / "nested" / "benchmark.json"
    arguments = ["train", "benchmark-processing-preparation", "--out", str(path),
                 "--proteins", "2", "--protein-length", "30", "--draws", "10", "--repeats", "1"]
    assert cli_main.main(arguments) == 0
    result = json.loads(path.read_text())
    assert result["sampler"]["exact_export_parity"] is True
    assert len(result["sampler"]["export_strings_seconds"]["seconds"]) == 1
    assert all(len(digest) == 64 for digest in result["source_hashes"].values())
    assert any(key.endswith("numeric_sequences.py") for key in result["source_hashes"])
    before = path.read_bytes()
    assert cli_main.main(arguments) != 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("relative_path", [
    "scripts/training/release_exact/make_train_data.processing.py",
    "downloads-generation/models_class1_processing/make_train_data.py",
])
def test_processing_generation_matches_and_preserves_scored_pool(relative_path, tmp_path, monkeypatch):
    import numpy
    import mhcflurry
    from mhcflurry.processing_matching import validate_matched_training_data
    module = load_script(REPO_ROOT / relative_path)
    args = module.parser.parse_args([
        "--hits", "unused", "--affinity-predictor", "frozen-public",
        "--proteome-peptides", "unused", "--ppv-multiplier", "3",
        "--out", str(tmp_path / "training.csv")])
    assert args.negative_policy == "matched"
    args.matching_reference = {"sha256": "a" * 64}
    directory = tmp_path / "training.csv.matching"
    directory.mkdir()
    hits = pandas.DataFrame({"peptide": ["SIINFEKL", "SIINFEKA"],
                             "sample_id": "s", "allele": "HLA-A*02:01",
                             "hit_id": [1, 2], "protein_accession": "p1",
                             "n_flank": "AAAAA", "c_flank": "CCCCC"})
    pool = pandas.DataFrame({"peptide": [c + "IINFEKL" for c in "CDEFGH"],
                             "protein_accession": "p1", "n_flank": "AAAAA", "c_flank": "CCCCC"})

    class Predictor:
        supported_alleles = ["HLA-A*02:01"]

        def canonicalize_allele_name(self, allele):
            return allele

        def predict(self, peptides, allele):
            assert allele == "HLA-A*02:01"
            return numpy.repeat(100.0, len(peptides))

    monkeypatch.setattr(mhcflurry.Class1AffinityPredictor, "load", lambda path: Predictor())
    result = module.do_process_samples(["s"], seed=42, constant_data={
        "args": args, "lengths": [8], "all_peptides_by_length": {8: pool},
        "sample_table": hits.drop_duplicates("sample_id").set_index("sample_id"), "hit_df": hits})
    validate_matched_training_data(result)
    assert result.hit.sum() == len(hits)
    assert len(result) == 4
    assert len(pandas.read_csv(next(directory.glob("*.candidate_pool.csv.bz2")))) == 8
    assert len(list(directory.glob("*.matching.json"))) == 1


def test_generate_scripts_keep_packaged_proteome_peptide_artifacts():
    """Guard download-generation artifact contracts.

    These GENERATE.sh scripts package their working directory into download
    bundles. The compressed proteome peptide CSVs are part of those bundles
    and must not disappear just because newer training paths can avoid reading
    them.
    """
    processing_generate = (
        REPO_ROOT / "downloads-generation/models_class1_processing/GENERATE.sh"
    ).read_text()
    assert "write_proteome_peptides.py" in processing_generate
    assert "--out \"$(pwd)/proteome_peptides.csv\"" in processing_generate
    assert "bzip2 -f proteome_peptides.csv" in processing_generate
    assert "--proteome-peptides \"$(pwd)/proteome_peptides.csv.bz2\"" in (
        processing_generate
    )

    predictions_generate = (
        REPO_ROOT / "downloads-generation/data_predictions/GENERATE.sh"
    ).read_text()
    assert "write_proteome_peptides.py" in predictions_generate
    assert "--out proteome_peptides.$subset.csv" in predictions_generate
    assert "bzip2 proteome_peptides.$subset.csv" in predictions_generate
    assert "proteome_peptides.$subset.csv.bz2" in predictions_generate


def test_processing_preparation_keeps_predictor_in_worker_and_resumes_without_loading(tmp_path, monkeypatch):
    import numpy
    import mhcflurry
    module = load_script(REPO_ROOT / "scripts/training/release_exact/make_train_data.processing.py")
    args = module.parser.parse_args([
        "--hits", "unused", "--affinity-predictor", "reference", "--proteome-peptides", "unused",
        "--ppv-multiplier", "3", "--out", str(tmp_path / "training.csv")])
    args.matching_reference = {"sha256": "a" * 64}
    (tmp_path / "training.csv.matching").mkdir()
    hits = pandas.DataFrame({"peptide": ["SIINFEKL", "SIINFEKA"] * 2,
        "sample_id": ["s1", "s1", "s2", "s2"], "allele": "HLA-A*02:01",
        "hit_id": [1, 2, 3, 4], "protein_accession": "p1", "n_flank": "AAAAA", "c_flank": "CCCCC"})
    pool = pandas.DataFrame({"peptide": [c + "IINFEKL" for c in "CDEFGH"],
                             "protein_accession": "p1", "n_flank": "AAAAA", "c_flank": "CCCCC"})
    loads = []
    predictions = []

    class Predictor:
        supported_alleles = ["HLA-A*02:01"]

        def canonicalize_allele_name(self, allele):
            return allele

        def predict(self, peptides, allele):
            predictions.append(list(peptides))
            return numpy.repeat(100.0, len(peptides))

    def load(path):
        loads.append(path)
        return Predictor()

    monkeypatch.setattr(mhcflurry.Class1AffinityPredictor, "load", load)
    context = {"args": args, "lengths": [8], "all_peptides_by_length": {8: pool},
               "sample_table": hits.drop_duplicates("sample_id").set_index("sample_id"), "hit_df": hits}
    first = [module.do_process_samples([sample], seed=42, constant_data=context) for sample in ["s1", "s2"]]
    assert len(loads) == 1
    assert len(predictions) == 2
    del context["processing_affinity_predictor"]
    for index, sample in enumerate(["s1", "s2"]):
        second = module.do_process_samples([sample], seed=42, constant_data=context)
        pandas.testing.assert_frame_equal(first[index], second, check_dtype=False)
    assert len(loads) == 1
    assert len(predictions) == 2


def test_numeric_pipeline_is_schedule_independent_and_resume_skips_scoring(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    import mhcflurry
    from mhcflurry.common import derive_seed
    from mhcflurry.numeric_sequences import NumericSequences
    from mhcflurry.processing_matching import validate_matched_training_data
    module = load_script(REPO_ROOT / "scripts/training/release_exact/make_train_data.processing.py")
    caller = threading.get_ident()
    calls = []

    class Predictor:
        supported_alleles = ["HLA-A*02:01"]
        class1_pan_allele_models = [SimpleNamespace(get_device=lambda: "cpu")]

        def canonicalize_allele_name(self, allele):
            return allele

        def predict_numeric(self, peptides, allele):
            assert threading.get_ident() == caller
            assert isinstance(peptides, NumericSequences)
            calls.append(len(peptides))
            return 100 + peptides.indices[:, :2].numpy().sum(axis=1).astype(float)

    loads = []

    def load(path):
        loads.append(path)
        return Predictor()

    monkeypatch.setattr(mhcflurry.Class1AffinityPredictor, "load", load)
    hits = pandas.DataFrame([dict(peptide=p, sample_id=s, hit_id=i * 2 + j,
        protein_accession="p", n_flank="XXXXX", c_flank="YYYYY")
        for i, s in enumerate(["s1", "s2", "s3", "s4"])
        for j, p in enumerate(["ACDEFGHI", "CDEFGHIKL"])])
    results = []
    for depth in (1, 3):
        out = tmp_path / str(depth)
        out.mkdir()
        args = module.parser.parse_args([
            "--hits", "unused", "--affinity-predictor", "reference", "--proteome-reference-csv", "unused",
            "--ppv-multiplier", "20", "--preparation-pipeline-depth", str(depth), "--out", str(out / "train.csv")])
        args.matching_reference = {"sha256": "a" * 64}
        (out / "train.csv.matching").mkdir()
        context = dict(args=args, lengths=[8, 9, 10, 11], hit_df=hits,
            proteome_sequences={"p": "ACDEFGHIKLMNPQRSTVWY" * 10}, flanking_length=5,
            sample_table=hits.drop_duplicates("sample_id").set_index("sample_id").assign(allele="HLA-A*02:01"),
            sample_seeds={s: derive_seed(42, "sample", s) for s in hits.sample_id.unique()})
        result = module.do_process_samples(hits.sample_id.unique(), constant_data=context)
        validate_matched_training_data(result)
        results.append(result)
        old_calls = list(calls)
        del context["processing_affinity_predictor"]
        resumed = module.do_process_samples(hits.sample_id.unique(), constant_data=context)
        pandas.testing.assert_frame_equal(result, resumed, check_dtype=False)
        assert calls == old_calls
    pandas.testing.assert_frame_equal(*results)
    assert len(loads) == 2  # One per worker context, no reload for any resumed sample.


def test_model_selection_decoys_do_not_require_unused_proteome_peptides():
    module = load_script(
        REPO_ROOT / "downloads-generation/analysis_predictor_info"
        / "generate_model_selection_with_decoys.py"
    )

    args = module.parser.parse_args([
        "model_selection.csv",
        "--protein-data", "proteins.csv",
        "--out", "with_decoys.csv",
    ])
    assert args.data == "model_selection.csv"
    assert args.protein_data == "proteins.csv"
    assert args.out == "with_decoys.csv"
    assert not hasattr(args, "proteome_peptides")

    generate = (
        REPO_ROOT / "downloads-generation/analysis_predictor_info/GENERATE.sh"
    ).read_text()
    assert "--proteome-peptides" not in generate


@pytest.mark.parametrize("relative_path", [
    "scripts/training/release_exact/make_train_data.processing.py",
    "downloads-generation/models_class1_processing/make_train_data.py",
])
def test_processing_train_data_canonicalizes_alleles_before_prediction(
        relative_path):
    module = load_script(REPO_ROOT / relative_path)

    assert module.canonicalize_processing_allele("HLA-A0201") == (
        "HLA-A*02:01"
    )
    assert module.canonicalize_processing_allele("B0702") == "HLA-B*07:02"
    assert module.canonicalize_processing_allele("HLA-C*03:04") == (
        "HLA-C*03:04"
    )
    assert module.canonicalize_processing_allele("HLA-B*44:01") == (
        "HLA-B*44:01"
    )
    assert module.canonicalize_processing_allele("H-2-Kb") is None
    assert module.canonicalize_processing_allele("NONSENSE") is None

    for ambiguous in ("A2", "HLA-A2", "A*02", "HLA-B15"):
        with pytest.raises(ValueError, match="sequence-resolved"):
            module.canonicalize_processing_allele(ambiguous)

    class FakePredictor:
        supported_alleles = ("HLA-A*02:01",)

        def __init__(self):
            self.seen = []

        def canonicalize_allele_name(self, allele):
            self.seen.append(allele)
            return allele

    predictor = FakePredictor()
    canonical = module.canonicalize_processing_allele("HLA-A0201")
    assert module.predictor_allele_for_processing(predictor, canonical) == (
        "HLA-A*02:01"
    )
    assert predictor.seen == ["HLA-A*02:01"]

    with pytest.raises(ValueError, match="does not support"):
        module.predictor_allele_for_processing(
            predictor, "HLA-B*07:02"
        )

    required = [
        "--hits", "hits.csv",
        "--affinity-predictor", "models",
        "--proteome-peptides", "proteome.csv",
        "--out", "out.csv",
    ]
    for option in ("--hit-multiplier-to-take", "--ppv-multiplier"):
        with pytest.raises(SystemExit):
            module.parser.parse_args(required + [option, "0"])


def test_presentation_train_data_canonicalizes_genotype_tokens():
    module = load_script(
        REPO_ROOT / "scripts/training/release_exact"
        / "make_train_data.presentation.py"
    )

    assert module.split_hla_genotype("A*02:01 HLA-B0702 H-2-Kb") == (
        "HLA-A*02:01",
        "HLA-B*07:02",
        "H2-K*b",
    )
    for invalid in ("HLA-A2", "A*02", "NONSENSE", float("nan"), ""):
        with pytest.raises(ValueError):
            module.split_hla_genotype(invalid)

    with pytest.raises(SystemExit):
        module.parser.parse_args([
            "--hits", "hits.csv",
            "--proteome-peptides", "proteome.csv",
            "--out", "out.csv",
            "--decoys-per-hit", "0",
        ])
    for fraction in ("0", "1.1"):
        with pytest.raises(SystemExit):
            module.parser.parse_args([
                "--hits", "hits.csv",
                "--proteome-peptides", "proteome.csv",
                "--out", "out.csv",
                "--sample-fraction", fraction,
            ])


def test_annotate_tpm_matches_rowwise_expression_sum():
    module = load_script(
        REPO_ROOT / "downloads-generation/models_class1_processing"
        / "annotate_hits_with_expression.py"
    )
    hit_df = pandas.DataFrame(
        {
            "protein_ensembl": ["gene1 gene2", "gene3", "missing gene2"],
            "expression_dataset": ["sample_a", "sample_b", "sample_a"],
        },
        index=[10, 20, 30],
    )
    expression_df = pandas.DataFrame(
        {
            "sample_a": [1.5, 2.0, 3.0],
            "sample_b": [10.0, 20.0, 30.0],
        },
        index=["gene1", "gene2", "gene3"],
    )

    result = module.annotate_tpm(hit_df, expression_df)

    pandas.testing.assert_series_equal(
        result,
        pandas.Series([3.5, 30.0, 2.0], index=[10, 20, 30]),
    )


def test_write_proteome_peptides_streams_expected_rows(tmp_path):
    input_csv = tmp_path / "annotated_ms.csv"
    reference_csv = tmp_path / "proteins.csv"
    out_csv = tmp_path / "proteome_peptides.csv"
    pandas.DataFrame(
        {
            "mhc_class": ["I", "I"],
            "protein_ensembl_primary": ["ENSG1", "ENSG2"],
            "n_flank": ["XX", "XX"],
            "protein_accession": ["P1", "P2"],
        }
    ).to_csv(input_csv, index=False)
    pandas.DataFrame(
        {
            "name": ["protein1", "protein2"],
            "accession": ["P1", "P2"],
            "seq": ["ACDEFGHIK", "ACDEFGHI"],
        }
    ).to_csv(reference_csv, index=False)

    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT / "downloads-generation/models_class1_processing"
                / "write_proteome_peptides.py"
            ),
            str(input_csv),
            str(reference_csv),
            "--out",
            str(out_csv),
            "--lengths",
            "8",
            "9",
        ],
        check=True,
    )

    result = pandas.read_csv(out_csv)

    assert result.to_dict("records") == [
        {
            "protein_accession": "P1",
            "peptide": "ACDEFGHI",
            "n_flank": "XX",
            "c_flank": "KX",
            "start_position": 0,
        },
        {
            "protein_accession": "P1",
            "peptide": "ACDEFGHIK",
            "n_flank": "XX",
            "c_flank": "XX",
            "start_position": 0,
        },
    ]

    sequences = load_reference_sequences(reference_csv, ["P1", "P2"])
    helper_result = make_peptide_frame_for_accessions(
        ["P1", "P2"],
        sequences,
        lengths=[8, 9],
        flanking_length=2,
    )
    pandas.testing.assert_frame_equal(result, helper_result)


def test_infer_flanking_length_rejects_mixed_lengths():
    hit_df = pandas.DataFrame({
        "n_flank": ["XX", "YYY"],
        "c_flank": ["XX", "YYY"],
    })

    try:
        infer_flanking_length(hit_df)
    except ValueError as e:
        assert "Expected one flank length" in str(e)
    else:
        raise AssertionError("Expected mixed flank lengths to fail")


def test_iter_protein_peptide_records_drops_cterminal_min_length_mer():
    # Pins the DELIBERATE off-by-one parity with the historical
    # write_proteome_peptides.py: range(0, len(seq) - min_length) has an
    # exclusive upper bound, so the min_length-mer starting at
    # len(seq) - min_length (the C-terminal min_length-mer) is NOT emitted.
    # This test exists so that behavior can't drift silently.
    seq = "ACDEFGHIKL"  # length 10
    records = list(iter_protein_peptide_records(
        "P1", seq, lengths=[8], flanking_length=2))
    starts = [start for (_, _, _, _, start) in records]
    peptides = [pep for (_, pep, _, _, _) in records]
    # Starts 0 and 1 only; start 2 (== len - min_length) is dropped.
    assert starts == [0, 1]
    assert peptides == ["ACDEFGHI", "CDEFGHIK"]
    assert "DEFGHIKL" not in peptides  # the dropped C-terminal 8-mer


def test_iter_protein_peptide_records_multi_length_counts():
    # With multiple lengths the only systematically dropped peptide is the
    # C-terminal min_length-mer; longer k-mers are bounded by the
    # end_pos > len(sequence) break, not the start range.
    seq = "ACDEFGHIKLMNPQR"  # length 15, min_length 8 -> starts 0..6
    records = list(iter_protein_peptide_records(
        "P1", seq, lengths=[8, 9], flanking_length=3))
    starts8 = [s for (_, pep, _, _, s) in records if len(pep) == 8]
    starts9 = [s for (_, pep, _, _, s) in records if len(pep) == 9]
    assert starts8 == list(range(0, 7))
    assert starts9 == list(range(0, 7))
    assert len(records) == 14
    # The 8-mer at start 7 (== len - min_length) is the dropped one.
    assert "IKLMNPQR" not in [pep for (_, pep, _, _, _) in records]


def test_sample_peptide_frame_for_accessions_excludes_peptides():
    sampled = sample_peptide_frame_for_accessions(
        ["P1"],
        {"P1": "ACDEFGHIKLMNPQRSTVWY"},
        lengths=[8],
        flanking_length=2,
        exclude_peptides={"ACDEFGHI"},
        n=3,
    )

    assert len(sampled) == 3
    assert "ACDEFGHI" not in set(sampled.peptide)
    assert set(sampled.columns) == {
        "protein_accession",
        "peptide",
        "n_flank",
        "c_flank",
        "start_position",
    }


def write_presentation_like_inputs(tmp_path):
    hits_csv = tmp_path / "hits.csv"
    reference_csv = tmp_path / "proteins.csv"
    pandas.DataFrame(
        {
            "hit_id": ["hit1"],
            "pmid": ["123"],
            "mhc_class": ["I"],
            "peptide": ["ACDEFGHI"],
            "protein_ensembl": ["ENSG1"],
            "hla": ["HLA-A*02:01 HLA-B*07:02"],
            "sample_id": ["sample1"],
            "format": ["MULTIALLELIC"],
            "protein_accession": ["P1"],
            "n_flank": ["XX"],
            "c_flank": ["XX"],
        }
    ).to_csv(hits_csv, index=False)
    pandas.DataFrame(
        {
            "name": ["protein1"],
            "accession": ["P1"],
            "seq": ["ACDEFGHIKLMNPQRSTVWY"],
        }
    ).to_csv(reference_csv, index=False)
    return hits_csv, reference_csv


def run_reference_decoy_script(script_path, tmp_path, random_seed=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hits_csv, reference_csv = write_presentation_like_inputs(tmp_path)
    out_csv = tmp_path / "train_data.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    command = [
            sys.executable,
            str(script_path),
            "--hits",
            str(hits_csv),
            "--proteome-reference-csv",
            str(reference_csv),
            "--decoys-per-hit",
            "4",
            "--only-format",
            "MULTIALLELIC",
            "--out",
            str(out_csv),
        ]
    if random_seed is not None:
        command.extend(["--random-seed", str(random_seed)])
    subprocess.run(
        command,
        env=env,
        check=True,
    )
    return pandas.read_csv(out_csv)


def assert_reference_decoy_output(result):
    assert len(result) == 5
    assert result.hit.value_counts().to_dict() == {0: 4, 1: 1}
    assert set(result.protein_accession) == {"P1"}
    assert set(result.hla) == {"HLA-A*02:01 HLA-B*07:02"}


def test_release_presentation_train_data_uses_reference_csv_decoys(tmp_path):
    result = run_reference_decoy_script(
        REPO_ROOT / "scripts/training/release_exact"
        / "make_train_data.presentation.py",
        tmp_path,
    )
    assert_reference_decoy_output(result)


def test_release_presentation_decoys_are_reproducible_from_seed(tmp_path):
    script = (
        REPO_ROOT / "scripts/training/release_exact"
        / "make_train_data.presentation.py"
    )
    first = run_reference_decoy_script(script, tmp_path / "first", 271)
    second = run_reference_decoy_script(script, tmp_path / "second", 271)

    pandas.testing.assert_frame_equal(first, second)


def test_download_presentation_train_data_uses_reference_csv_decoys(tmp_path):
    result = run_reference_decoy_script(
        REPO_ROOT / "downloads-generation/models_class1_presentation"
        / "make_train_data.py",
        tmp_path,
    )
    assert_reference_decoy_output(result)


def test_data_evaluation_benchmark_uses_reference_csv_decoys(tmp_path):
    result = run_reference_decoy_script(
        REPO_ROOT / "downloads-generation/data_evaluation"
        / "make_benchmark.py",
        tmp_path,
    )
    assert_reference_decoy_output(result)
