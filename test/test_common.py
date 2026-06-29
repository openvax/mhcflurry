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

"""Tests for common helpers."""

import logging

from mhcflurry.common import (
    filter_canonicalizable_alleles,
    AlleleKeyResolver,
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


def test_allele_key_resolver_priority():
    resolver = AlleleKeyResolver({"HLA-B*44:01", "HLA-A*02:01"})
    # An allele with its own key keeps that key (not remapped to B*44:02).
    assert resolver.resolve("HLA-B*44:01") == "HLA-B*44:01"
    # An alternative spelling normalizes to the key.
    assert resolver.resolve("HLA-A0201") == "HLA-A*02:01"
    # The modern alias of a retired key routes back to the key in the set.
    assert resolver.resolve("HLA-B*44:02") == "HLA-B*44:01"


def test_allele_key_resolver_raises_on_junk():
    import pytest
    with pytest.raises(ValueError):
        AlleleKeyResolver(set()).resolve("NONSENSE", raise_on_error=True)
    assert AlleleKeyResolver(set()).resolve(
        "NONSENSE", raise_on_error=False) is None


def test_allele_key_resolver_best_effort_vs_strict():
    resolver = AlleleKeyResolver({"HLA-A*02:01"})
    # Supported allele: both modes return the key.
    assert resolver.resolve("HLA-A0201") == "HLA-A*02:01"
    assert resolver.resolve_to_key("HLA-A0201") == "HLA-A*02:01"
    # A valid allele that is NOT a key: best-effort returns the normalized name
    # (so the caller can flag it as unsupported); strict returns None (drop it).
    assert resolver.resolve("HLA-B*07:02") == "HLA-B*07:02"
    assert resolver.resolve_to_key("HLA-B*07:02") is None
    # Unparseable: both return None (strict never raises).
    assert resolver.resolve("NONSENSE") is None
    assert resolver.resolve_to_key("NONSENSE") is None


def test_allele_key_resolver_empty_keys_skip_membership_steps():
    # With no keys, steps 1-2 are skipped; best-effort returns the normalized
    # name and strict returns None (nothing can match an empty key set).
    resolver = AlleleKeyResolver(set())
    assert resolver.resolve("HLA-A*02:01") == "HLA-A*02:01"
    assert resolver.resolve_to_key("HLA-A*02:01") is None


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
