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

"""Tests for pseudosequence filename registry."""
from mhcflurry.pseudosequences import (
    LEGACY_ALLELE_SEQUENCES_FILENAME,
    main,
    pseudosequence_filename_for_length,
    pseudosequence_filename_for_mapping,
    pseudosequence_path,
)


def test_pseudosequence_filename_for_length():
    assert (
        pseudosequence_filename_for_length(34)
        == "pseudosequences.netmhcpan.34aa.csv"
    )
    assert (
        pseudosequence_filename_for_length(37)
        == "pseudosequences.mhcflurry.37aa.csv"
    )
    assert (
        pseudosequence_filename_for_length(39)
        == "pseudosequences.mhcflurry.39aa.csv"
    )
    assert pseudosequence_filename_for_length(None) is None


def test_pseudosequence_filename_for_ambiguous_mapping():
    assert pseudosequence_filename_for_mapping({}) is None
    assert pseudosequence_filename_for_mapping({"HLA-A*02:01": None}) is None
    assert pseudosequence_filename_for_mapping({
        "HLA-A*02:01": "A" * 34,
        "HLA-A*03:01": "A" * 39,
    }) is None


def test_pseudosequence_path_prefers_canonical(tmp_path):
    canonical = tmp_path / "pseudosequences.mhcflurry.39aa.csv"
    canonical.write_text("allele,pseudosequence\n")

    assert pseudosequence_path(str(tmp_path), 39) == str(canonical)


def test_pseudosequence_path_falls_back_to_legacy(tmp_path):
    assert pseudosequence_path(str(tmp_path), 39) == str(
        tmp_path / LEGACY_ALLELE_SEQUENCES_FILENAME)


def test_pseudosequence_cli_filename(capsys):
    main(["filename", "--length", "39"])

    assert capsys.readouterr().out == "pseudosequences.mhcflurry.39aa.csv\n"
