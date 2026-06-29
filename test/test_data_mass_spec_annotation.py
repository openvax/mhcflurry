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

"""Tests for data_mass_spec_annotated generation helpers."""

import importlib.util
from pathlib import Path

import pandas
import pytest

from mhcflurry import peptide_reference


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


def sorted_annotation_rows(data):
    df = pandas.read_csv(data) if isinstance(data, Path) else data
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


def read_reference_dataframe(reference):
    return pandas.read_csv(reference, index_col=0)


def annotate_with_backend(peptides, reference, backend, flanking_length=2):
    peptide_df = pandas.read_csv(peptides)
    peptide_df["hit_id"] = "hit." + peptide_df.index.map("{0:07d}".format)
    peptide_df = peptide_df.set_index("hit_id")
    reference_df = read_reference_dataframe(reference)
    return peptide_reference._annotate_peptide_references(
        peptide_df,
        reference_df,
        flanking_length=flanking_length,
        backend=backend)


def test_annotation_script_uses_library_default(tmp_path):
    annotate = load_annotate_module()
    peptides, reference = write_inputs(tmp_path)
    out = tmp_path / "annotated.csv"

    annotate.run([
        str(peptides),
        str(reference),
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


def test_annotation_cli_does_not_expose_backend(tmp_path):
    annotate = load_annotate_module()
    peptides, reference = write_inputs(tmp_path)

    with pytest.raises(SystemExit):
        annotate.run([
            str(peptides),
            str(reference),
            "--backend", "sliding",
            "--flanking-length", "2",
            "--out", str(tmp_path / "out.csv"),
        ])


def test_annotation_helper_default_matches_sliding(tmp_path):
    peptides, reference = write_inputs(tmp_path)

    peptide_df = pandas.read_csv(peptides)
    peptide_df["hit_id"] = "hit." + peptide_df.index.map("{0:07d}".format)
    peptide_df = peptide_df.set_index("hit_id")
    reference_df = read_reference_dataframe(reference)

    sliding = annotate_with_backend(peptides, reference, "sliding")
    auto = peptide_reference.annotate_peptide_references(
        peptide_df,
        reference_df,
        flanking_length=2)

    pandas.testing.assert_frame_equal(
        sorted_annotation_rows(sliding),
        sorted_annotation_rows(auto),
        check_dtype=False)


def test_annotation_helper_default_matches_ahocorasick(tmp_path):
    peptides, reference = write_inputs(tmp_path)

    peptide_df = pandas.read_csv(peptides)
    peptide_df["hit_id"] = "hit." + peptide_df.index.map("{0:07d}".format)
    peptide_df = peptide_df.set_index("hit_id")
    reference_df = read_reference_dataframe(reference)

    ahocorasick = annotate_with_backend(peptides, reference, "ahocorasick")
    auto = peptide_reference.annotate_peptide_references(
        peptide_df,
        reference_df,
        flanking_length=2)

    pandas.testing.assert_frame_equal(
        sorted_annotation_rows(ahocorasick),
        sorted_annotation_rows(auto),
        check_dtype=False)


def test_annotation_ahocorasick_matches_sliding(tmp_path):
    peptides, reference = write_inputs(tmp_path)
    sliding = annotate_with_backend(peptides, reference, "sliding")
    ahocorasick = annotate_with_backend(peptides, reference, "ahocorasick")

    pandas.testing.assert_frame_equal(
        sorted_annotation_rows(sliding),
        sorted_annotation_rows(ahocorasick),
        check_dtype=False)
