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

"""Regression tests for Biopython-free FASTA handling in downloads-generation (#501)."""

import gzip
import importlib.util
import sys
import types
from pathlib import Path

import pandas

from mhcflurry.fasta import FastaParser


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative_path, module_name):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_sequences_and_make_allele_sequences_roundtrip(tmp_path):
    """filter_sequences.py -> make_allele_sequences.py works with mhcflurry.fasta."""
    filter_mod = _load_script(
        "downloads-generation/allele_sequences/filter_sequences.py",
        "filter_sequences_test_mod",
    )
    make_mod = _load_script(
        "downloads-generation/allele_sequences/make_allele_sequences.py",
        "make_allele_sequences_test_mod",
    )

    seq_a_short = "A" * 60
    seq_a_long = "M" * 10 + "Y" * 10 + "F" * 10 + "E" * 40  # 70 aa (replaces short)
    seq_b = "W" * 10 + "K" * 10 + "R" * 10 + "D" * 40  # 70 aa
    seq_ld = "G" * 10 + "H" * 10 + "I" * 10 + "L" * 40  # 70 aa

    raw_fasta = tmp_path / "raw.fasta"
    raw_fasta.write_text(
        "\n".join([
            ">HLA:HLA00001 A*01:01:01:01 60 bp",
            seq_a_short,
            ">HLA:HLA00002 A*01:01:01:02 70 bp",
            seq_a_long,
            ">HLA:HLA00003 B*07:02:01 70 bp",
            seq_b,
            ">HLA:SHORT A*02:01:01 10 bp",
            "ACDEFGHIKL",
            ">sp|P01897|HA1L_MOUSE H-2 class I histocompatibility antigen, L-D alpha chain OS=Mus musculus",
            seq_ld,
            ">UNPARSEABLE not_an_mhc_allele",
            "Q" * 70,
        ]) + "\n",
        encoding="utf-8",
    )

    filtered_fasta = tmp_path / "filtered.fasta"
    filter_mod.run([str(raw_fasta), "--out", str(filtered_fasta)])

    parsed = list(
        FastaParser().iterate_over_file(filtered_fasta, full_descriptions=True)
    )
    # Preserves first-seen order (A*01:01, B*07:02, H2-L*d), drops <50 aa and unparseable,
    # replaces A*01:01 with the longer 70-aa sequence, and writes Biopython-compatible
    # `>{record_id} {normalized_name} {description}` headers so split()[1] is the allele.
    assert [desc.split()[1] for desc, _ in parsed] == [
        "HLA-A*01:01",
        "HLA-B*07:02",
        "H2-L*d",
    ]
    assert [seq for _, seq in parsed] == [seq_a_long, seq_b, seq_ld]
    # Check 60-char wrapping in output file.
    seq_lines = [
        line for line in filtered_fasta.read_text(encoding="utf-8").splitlines()
        if not line.startswith(">")
    ]
    assert max(len(line) for line in seq_lines) == 60

    # Feed the filtered (equal-length, aligned) FASTA straight into make_allele_sequences.py.
    recap_csv = tmp_path / "recap.csv"
    pandas.DataFrame({
        "allele": ["HLA-A*01:01", "HLA-B*07:02", "H2-L*d"],
        "sequence": [
            seq_a_long[0] + seq_a_long[15] + seq_a_long[25],
            seq_b[0] + seq_b[15] + seq_b[25],
            seq_ld[0] + seq_ld[15] + seq_ld[25],
        ],
    }).to_csv(recap_csv, index=False)

    out_csv = tmp_path / "allele_sequences.csv"
    make_mod.run([
        str(filtered_fasta),
        "--recapitulate-sequences", str(recap_csv),
        "--out-csv", str(out_csv),
    ])
    out_df = pandas.read_csv(out_csv)
    assert list(out_df["normalized_allele"]) == [
        "HLA-A*01:01",
        "HLA-B*07:02",
        "H2-L*d",
    ]
    assert list(out_df["sequence"]) == ["MYF", "WKR", "GHI"]


def test_data_references_process_reads_gzip_fasta_without_biopython(
        tmp_path, monkeypatch):
    """downloads-generation/data_references/process.py parses .fasta.gz via mhcflurry.fasta."""
    pushed = []
    written_index = {}

    class DummyFMIndex:
        def push_back(self, value):
            pushed.append(value)

        def build(self):
            written_index["built"] = True

        def write(self, path):
            written_index["path"] = path
            Path(path).write_text("dummy-fm-index", encoding="utf-8")

    dummy_shellinford = types.SimpleNamespace(FMIndex=DummyFMIndex)
    dummy_gtfparse = types.SimpleNamespace(
        read_gtf=lambda path: pandas.DataFrame({
            "feature": ["gene"],
            "gene_id": ["ENSG000001"],
            "seqname": ["chr1"],
            "start": [100],
            "end": [500],
            "strand": ["+"],
        })
    )
    monkeypatch.setitem(sys.modules, "shellinford", dummy_shellinford)
    monkeypatch.setitem(sys.modules, "gtfparse", dummy_gtfparse)

    process_mod = _load_script(
        "downloads-generation/data_references/process.py",
        "data_references_process_test_mod",
    )

    fasta_gz = tmp_path / "proteome.fasta.gz"
    with gzip.open(fasta_gz, "wt", encoding="utf-8") as fd:
        fd.write(">sp|P12345|PROT_HUMAN Example protein\nacdef\nghijk\n")

    id_mapping = tmp_path / "idmapping.gz"
    with gzip.open(id_mapping, "wt", encoding="utf-8") as fd:
        fd.write("P12345\tEnsembl\tENSG000001\n")
        fd.write("P12345\tEnsembl_TRS\tENST000001\n")
        fd.write("P12345\tGene_Name\tTP53\n")

    gtf_path = tmp_path / "dummy.gtf.gz"
    gtf_path.write_bytes(b"")

    out_csv = tmp_path / "uniprot_proteins.csv"
    out_index = tmp_path / "uniprot_proteins.fm"
    process_mod.run([
        str(fasta_gz),
        "--id-mapping", str(id_mapping),
        "--ensembl-gtf", str(gtf_path),
        "--out-csv", str(out_csv),
        "--out-index", str(out_index),
    ])

    df = pandas.read_csv(out_csv, index_col=0)
    assert list(df["name"]) == ["sp|P12345|PROT_HUMAN"]
    assert list(df["description"]) == ["sp|P12345|PROT_HUMAN Example protein"]
    assert list(df["seq"]) == ["ACDEFGHIJK"]
    assert list(df["db"]) == ["sp"]
    assert list(df["accession"]) == ["P12345"]
    assert list(df["entry"]) == ["PROT_HUMAN"]
    assert list(df["ensembl_primary"]) == ["ENSG000001"]
    assert list(df["primary_ensembl_contig"]) == ["chr1"]
    assert pushed == ["$ACDEFGHIJK$"]
    assert written_index == {"built": True, "path": str(out_index)}
