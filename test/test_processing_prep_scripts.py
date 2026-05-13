import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas

from mhcflurry.proteome_decoys import (
    infer_flanking_length,
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


def run_reference_decoy_script(script_path, tmp_path):
    hits_csv, reference_csv = write_presentation_like_inputs(tmp_path)
    out_csv = tmp_path / "train_data.csv"
    subprocess.run(
        [
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
        ],
        check=True,
    )
    return pandas.read_csv(out_csv)


def assert_reference_decoy_output(result):
    assert len(result) == 5
    assert result.hit.value_counts().to_dict() == {0: 4, 1: 1}
    assert set(result.protein_accession) == {"P1"}


def test_release_presentation_train_data_uses_reference_csv_decoys(tmp_path):
    result = run_reference_decoy_script(
        REPO_ROOT / "scripts/training/release_exact"
        / "make_train_data.presentation.py",
        tmp_path,
    )
    assert_reference_decoy_output(result)


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
