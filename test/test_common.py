"""Tests for common helpers."""

import logging

from mhcflurry.common import filter_canonicalizable_alleles


def test_filter_canonicalizable_alleles_logs_instead_of_stdout(
        caplog, capsys):
    with caplog.at_level(logging.WARNING):
        result = filter_canonicalizable_alleles(
            ["HLA-A*02:01", "HLA-A*02:01N"],
            log_label="test alleles",
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert result == ["HLA-A*02:01"]
    assert "Skipping 1 test alleles" in caplog.text
