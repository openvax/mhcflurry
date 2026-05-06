"""Tests for pseudosequence filename registry."""
from mhcflurry.pseudosequences import (
    LEGACY_ALLELE_SEQUENCES_FILENAME,
    main,
    pseudosequence_filename_for_length,
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
