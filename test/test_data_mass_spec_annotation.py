"""Tests for data_mass_spec_annotated generation helpers."""

import importlib.util
from pathlib import Path

import pandas
import pytest


ANNOTATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "downloads-generation"
    / "data_mass_spec_annotated"
    / "annotate.py"
)


def load_annotate_module():
    spec = importlib.util.spec_from_file_location(
        "data_mass_spec_annotated_annotate",
        ANNOTATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sorted_annotation_rows(path):
    df = pandas.read_csv(path)
    cols = [
        "hit_id",
        "peptide",
        "protein_accession",
        "start_position",
        "num_proteins",
        "num_occurrences_in_protein",
        "n_flank",
        "c_flank",
        "sample_id",
    ]
    return df[cols].sort_values(cols).reset_index(drop=True)


def write_inputs(tmp_path):
    peptides = tmp_path / "peptides.csv"
    pandas.DataFrame({
        "peptide": ["AAA", "AAA", "CDE"],
        "sample_id": ["s1", "s2", "s3"],
    }).to_csv(peptides, index=False)

    reference = tmp_path / "reference.csv"
    pandas.DataFrame({
        "accession": ["P1", "P2", "P3"],
        "name": ["protein1", "protein2", "protein3"],
        "description": ["first", "second", "overlap"],
        "seq": ["MMAAAKAAAQCDE", "CDEAAA", "AAAAA"],
        "db": ["sp", "tr", "tr"],
        "gene_name": ["G1", "G2", "G3"],
    }).to_csv(reference)
    return peptides, reference


def test_annotation_sliding_backend(tmp_path):
    annotate = load_annotate_module()
    peptides, reference = write_inputs(tmp_path)
    out = tmp_path / "annotated.csv"

    annotate.run([
        str(peptides),
        str(reference),
        "--backend", "sliding",
        "--flanking-length", "2",
        "--out", str(out),
    ])

    result = sorted_annotation_rows(out)
    assert len(result) == 10

    aaa_rows = result.loc[result.peptide == "AAA"]
    assert set(aaa_rows.num_proteins) == {3}
    p1_rows = aaa_rows.loc[aaa_rows.protein_accession == "P1"]
    assert set(p1_rows.num_occurrences_in_protein) == {2}
    assert set(p1_rows.start_position) == {2, 6}

    # Preserve legacy re.finditer-style non-overlapping occurrence semantics.
    p3_rows = aaa_rows.loc[aaa_rows.protein_accession == "P3"]
    assert set(p3_rows.start_position) == {0}
    assert set(p3_rows.num_occurrences_in_protein) == {1}


def test_annotation_auto_matches_sliding(tmp_path):
    annotate = load_annotate_module()
    peptides, reference = write_inputs(tmp_path)
    sliding = tmp_path / "sliding.csv"
    auto = tmp_path / "auto.csv"

    common = [
        str(peptides),
        str(reference),
        "--flanking-length", "2",
    ]
    annotate.run(common + ["--backend", "sliding", "--out", str(sliding)])
    annotate.run(common + ["--backend", "auto", "--out", str(auto)])

    pandas.testing.assert_frame_equal(
        sorted_annotation_rows(sliding),
        sorted_annotation_rows(auto),
        check_dtype=False)


def test_annotation_ahocorasick_matches_sliding_when_available(tmp_path):
    pytest.importorskip("ahocorasick_rs")
    annotate = load_annotate_module()
    peptides, reference = write_inputs(tmp_path)
    sliding = tmp_path / "sliding.csv"
    ahocorasick = tmp_path / "ahocorasick.csv"

    common = [
        str(peptides),
        str(reference),
        "--flanking-length", "2",
    ]
    annotate.run(common + ["--backend", "sliding", "--out", str(sliding)])
    annotate.run(
        common + ["--backend", "ahocorasick", "--out", str(ahocorasick)])

    pandas.testing.assert_frame_equal(
        sorted_annotation_rows(sliding),
        sorted_annotation_rows(ahocorasick),
        check_dtype=False)
