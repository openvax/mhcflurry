"""Tests for common helpers."""

import logging

from mhcflurry.common import (
    filter_canonicalizable_alleles,
    build_allele_alias_map,
    canonicalize_allele_to_keys,
    canonicalize_allele_series,
)


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


def test_filter_canonicalizable_alleles_returns_names_verbatim():
    # Survivors must be returned unchanged — NOT alias-remapped — so a retired
    # name stored as its own pseudosequence key is not collapsed onto its alias
    # target (which would silently drop it on the default calibrate path).
    assert filter_canonicalizable_alleles(["HLA-B*44:01", "JUNK"]) == \
        ["HLA-B*44:01"]


def test_canonicalize_allele_to_keys_no_alias_first():
    keys = {"HLA-B*44:01", "HLA-A*02:01"}
    alias_map = build_allele_alias_map(keys)
    # An allele with its own key keeps that key (not remapped to B*44:02).
    assert canonicalize_allele_to_keys(
        "HLA-B*44:01", keys, alias_map) == "HLA-B*44:01"
    # An alternative spelling normalizes to the key.
    assert canonicalize_allele_to_keys(
        "HLA-A0201", keys, alias_map) == "HLA-A*02:01"
    # The modern alias of a retired key routes back to the key in the set.
    assert canonicalize_allele_to_keys(
        "HLA-B*44:02", keys, alias_map) == "HLA-B*44:01"


def test_canonicalize_allele_to_keys_raises_on_junk():
    import pytest
    with pytest.raises(ValueError):
        canonicalize_allele_to_keys("NONSENSE", set(), {}, raise_on_error=True)
    assert canonicalize_allele_to_keys(
        "NONSENSE", set(), {}, raise_on_error=False) is None


def test_canonicalize_allele_series_resolves_aliases_and_drops_junk(caplog):
    keys = ["HLA-B*44:01", "HLA-A*02:01"]
    with caplog.at_level(logging.WARNING):
        out = canonicalize_allele_series(
            ["HLA-B*44:01", "HLA-A0201", "HLA-B*44:02", "NONSENSE"],
            keys,
            log_label="training alleles")
    # Aliases / spellings resolve to keys; unparseable -> None (dropped).
    assert out == ["HLA-B*44:01", "HLA-A*02:01", "HLA-B*44:01", None]
    # Every non-None result is a member of the key set.
    assert all(x in set(keys) for x in out if x is not None)
    assert "Dropping 1 training alleles" in caplog.text
