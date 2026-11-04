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

"""Run the historical MS curation builder on small in-memory workbooks.

``downloads-generation/data_curated/curate_ms_by_pmid.py`` used pandas APIs
that pandas 3 removed (``DataFrame.applymap``, ``fillna(method=...)``,
positional ``axis``; #488). These tests run the PMID 31844290 handler and the
script's ``run()`` normalization without the source workbooks.
"""

import importlib.util
from pathlib import Path
import sys

import pandas
import pytest


CURATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "downloads-generation"
    / "data_curated"
    / "curate_ms_by_pmid.py"
)


def load_curate_module():
    """
    Import ``curate_ms_by_pmid.py`` from its path as a fresh module.

    Returns
    -------
    module
    """
    spec = importlib.util.spec_from_file_location(
        "curate_ms_by_pmid", CURATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keskin_workbooks():
    """
    Build stand-ins for the two PMID 31844290 supplementary workbooks.

    Returns
    -------
    dict
        Workbook file name mapped to ``{sheet name: DataFrame}``, as
        ``pandas.read_excel(..., sheet_name=None)`` returns it.
    """
    mono = {
        "Sheet1": pandas.DataFrame({"ignored": [1]}),
        "A0201": pandas.DataFrame({"sequence": ["siinfekl", "NLVPMVATV"]}),
    }
    metadata = pandas.DataFrame({
        "Clinical ID": ["CLL-1", "CLL-2", None, "Mel-3"],
        # CLL-2's blank cells are merged-cell continuations of CLL-1's.
        "Cancer type": ["CLL", None, None, "Melanoma"],
        "IP Ab": ["W6/32", None, None, "W6/32"],
        "HLA-A": ["A0201", "A0301", "A0101", "n.d."],
        "HLA-B": ["B51", "B0702", "B0801", "n.d."],
        "HLA-C": ["-C0702", "C0401", "C0701", "n.d."],
    })
    peptides = pandas.DataFrame({
        "directory": ["CLL-1_rep1", "CLL-1_rep1", "CLL-2_rep1", "Pat7_rep1"],
        "sequence": ["siinfekl", "AAAWYLWEV", "KLGGALQAK", "GILGFVFTL"],
    })
    multi = {
        "Tissue Sample Characteristics": metadata,
        "Peptides": peptides,
    }
    return {"mono.xlsx": mono, "multi.xlsx": multi}


@pytest.fixture
def curate(monkeypatch):
    """The curation module, with ``pandas.read_excel`` serving
    :func:`keskin_workbooks`."""
    workbooks = keskin_workbooks()

    def fake_read_excel(filename, sheet_name=None):
        assert sheet_name is None
        return workbooks[Path(filename).name]

    monkeypatch.setattr(pandas, "read_excel", fake_read_excel)
    return load_curate_module()


def test_keskin_handler(curate):
    result = curate.PMID_HANDLERS["31844290"]("multi.xlsx", "mono.xlsx")
    by_sample = result.groupby("sample_id")

    assert sorted(by_sample.groups) == [
        "keskin_A0201", "keskin_CLL-1", "keskin_CLL-2"]
    assert result.peptide.tolist() == [
        "SIINFEKL", "NLVPMVATV", "SIINFEKL", "AAAWYLWEV", "KLGGALQAK"]

    first = by_sample.first()
    assert first.loc["keskin_A0201", "hla"] == "HLA-A*02:01"
    assert first.loc["keskin_A0201", "format"] == "monoallelic"
    # Leading "-" stripped and "B51" read as B*51:01 before normalization.
    assert first.loc["keskin_CLL-1", "hla"] == (
        "HLA-A*02:01 HLA-B*51:01 HLA-C*07:02")
    # CLL-2's cancer type and antibody are forward-filled from CLL-1.
    assert first.loc["keskin_CLL-2", "sample_type"] == "B-CELL"
    assert first.loc["keskin_CLL-2", "pulldown_antibody"] == "W6/32"
    assert first.loc["keskin_CLL-2", "format"] == "multiallelic"


def test_run_normalizes_ms_table(curate, monkeypatch, tmp_path):
    def expression_handler(*filenames):
        df = pandas.DataFrame(
            {"s1": [1.0, 3.0], "s2": [3.0, 5.0]}, index=["GENE1", "GENE2"])
        return [curate.make_expression_groups(
            "fake", df, {"sample_type:B-CELL": ["s1", "s2"]})]

    def spaced_handler(*filenames):
        return pandas.DataFrame({
            "sample_id": ["extra sample"],
            "peptide": ["gilgfvftl"],
            "format": ["multiallelic"],
            "mhc_class": ["I"],
            "hla": ["HLA-A*02:01   HLA-B*07:02 "],
            "cell_line": [None],
            "sample_type": ["B-CELL"],
        })

    monkeypatch.setitem(curate.EXPRESSION_HANDLERS, "fake", expression_handler)
    monkeypatch.setitem(curate.PMID_HANDLERS, "1", spaced_handler)
    monkeypatch.setattr(curate, "CELL_LINE_MIXTURES", [])
    ms_out = tmp_path / "ms.csv"
    expression_out = tmp_path / "expression.csv"
    monkeypatch.setattr(sys, "argv", [
        "curate_ms_by_pmid.py",
        "--expression-item", "fake", "expression.tsv",
        "--ms-item", "31844290", "mono.xlsx", "multi.xlsx",
        "--ms-item", "1", "extra.csv",
        "--ms-out", str(ms_out),
        "--expression-out", str(expression_out),
    ])

    curate.run()

    ms_df = pandas.read_csv(ms_out, keep_default_na=False)
    assert ms_df.columns[:7].tolist() == [
        "pmid", "sample_id", "peptide", "format", "mhc_class", "hla",
        "expression_dataset"]
    assert set(ms_df.expression_dataset) == {"sample_type:B-CELL"}
    # Every cell is upper-cased text, including the missing cell line.
    assert set(ms_df.format) == {"MONOALLELIC", "MULTIALLELIC"}
    extra = ms_df.loc[ms_df.pmid == 1].iloc[0]
    assert extra.sample_id == "EXTRASAMPLE"
    assert extra.peptide == "GILGFVFTL"
    assert extra.hla == "HLA-A*02:01 HLA-B*07:02"
    assert extra.cell_line == "NONE"

    expression = pandas.read_csv(expression_out, index_col=0)
    assert expression["sample_type:B-CELL"].tolist() == [2.0, 4.0]
