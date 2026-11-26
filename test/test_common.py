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

import argparse
import logging

import pytest

from mhcflurry.common import (
    allele_locus_name,
    filter_canonicalizable_alleles,
    AlleleKeyResolver,
    canonicalize_allele_series,
    fraction_arg,
    normalize_allele_name,
    normalize_class1_genotype,
    normalize_sequence_resolved_allele_name,
    positive_float_arg,
    positive_int_arg,
)


@pytest.mark.parametrize(
    "value", ["0", "-1", "1.5", "junk", 1.5, True]
)
def test_positive_int_arg_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int_arg(value)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "junk"])
def test_positive_float_arg_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float_arg(value)


def test_positive_arg_helpers_accept_values():
    assert positive_int_arg("12") == 12
    assert positive_float_arg("1e-3") == 1e-3


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "junk"])
def test_fraction_arg_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        fraction_arg(value)


def test_fraction_arg_accepts_fraction_and_one():
    assert fraction_arg("0.25") == 0.25
    assert fraction_arg("1") == 1.0


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


def test_allele_locus_name_uses_mhcgnomes():
    assert allele_locus_name("HLA-A0201") == "HLA-A"
    assert allele_locus_name("HLA-B*07:02") == "HLA-B"
    assert allele_locus_name("HLA-C*03:04") == "HLA-C"
    assert allele_locus_name("HLA-A2") == "HLA-A"
    assert allele_locus_name("HLA-B15") == "HLA-B"
    assert allele_locus_name("A2") == "HLA-A"
    assert allele_locus_name("H-2-Kb") == "H2"
    assert allele_locus_name("NONSENSE") == "other"


def test_normalize_allele_name_uses_exact_gene_classification():
    with pytest.raises(ValueError, match="MIC-family"):
        normalize_allele_name("HLA-MICA*001")
    with pytest.raises(ValueError, match="HFE-family"):
        normalize_allele_name("HLA-HFE*01:01")
    # A forbidden token appearing only as part of an unrelated gene name must
    # not trigger legacy substring inference.
    assert normalize_allele_name(
        "HLA-A*02:01", forbidden_substrings=("A0",)
    ) == "HLA-A*02:01"


def test_sequence_resolved_allele_and_genotype_normalization():
    assert normalize_sequence_resolved_allele_name("A0201") == "HLA-A*02:01"
    assert normalize_sequence_resolved_allele_name("H-2-Kb") == "H2-K*b"
    assert (
        normalize_sequence_resolved_allele_name("DLA-88*01:01")
        == "DLA-88*001:01"
    )
    assert normalize_class1_genotype("B0702 A0201 A0201") == (
        "HLA-B*07:02", "HLA-A*02:01")
    for value in ("HLA-A2", "A*02", "HLA-DQA1*01:01", "NONSENSE"):
        with pytest.raises(ValueError):
            normalize_sequence_resolved_allele_name(value)
    for value in (None, float("nan"), ""):
        with pytest.raises(ValueError, match="non-empty"):
            normalize_class1_genotype(value)


def test_allele_key_resolver_priority():
    resolver = AlleleKeyResolver({"HLA-B*44:01", "HLA-A*02:01"})
    # An allele with its own key keeps that key (not remapped to B*44:02).
    assert resolver.resolve("HLA-B*44:01") == "HLA-B*44:01"
    # An alternative spelling normalizes to the key.
    assert resolver.resolve("HLA-A0201") == "HLA-A*02:01"
    # The modern alias of a retired key routes back to the key in the set.
    assert resolver.resolve("HLA-B*44:02") == "HLA-B*44:01"


def test_allele_key_resolver_raises_on_junk():
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


@pytest.mark.parametrize(("raw_name", "message"), [
    ("HLA-DQA1*01:01", "MHC class II allele"),
    ("HLA-A2", "serotype is not a specific allele"),
    ("Caja-B5*01:01ps", "pseudogene MHC allele"),
    ("HLA-H*02:01", "pseudogene MHC allele"),
    ("Mamu-G*01:01", "pseudogene MHC allele"),
    ("Popy-Ap*01:01", "pseudogene MHC allele"),
    ("HLA-A*02:01N", "null-expression MHC allele"),
    ("HLA-A*02:01Q", "questionable-expression MHC allele"),
    ("MICA*001:01", "Unsupported gene in MHC allele name"),
    ("HFE*01:01", "Unsupported gene in MHC allele name"),
    ("MIC-NONSENSE", "Invalid MHC allele name"),
    ("NONSENSE", "Invalid MHC allele name"),
])
def test_normalize_allele_name_reports_specific_failure(raw_name, message):
    with pytest.raises(ValueError, match=message):
        normalize_allele_name(raw_name)
    assert normalize_allele_name(
        raw_name, raise_on_error=False, default_value="invalid") == "invalid"


def test_normalize_allele_name_scopes_inherited_pseudogene_status():
    # mhcgnomes marks Popy-Ap as a pseudogene but does not leak that property
    # to the similarly named Poab-Ap locus in a sibling orangutan species.
    assert normalize_allele_name("Poab-Ap*01:01") == "Poab-Ap*01:01"


def test_normalize_allele_name_accepts_class1_accessioned_alleles():
    # Sequence-backed AccessionedAllele identifiers (mhcgnomes >= 3.84 /
    # mhcseqs >= 2.14) canonicalize cleanly for both gene-qualified and
    # gene-less Class I alleles while rejecting Class II and forbidden genes.
    assert (
        normalize_allele_name("TriakisScyllium-UAA*uniprot:O46817")
        == "TriakisScyllium-UAA*uniprot:O46817"
    )
    assert (
        normalize_allele_name("OryziasLatipes-UBA*uniprot:H2LR16")
        == "OryziasLatipes-UBA*uniprot:H2LR16"
    )
    assert (
        normalize_allele_name("SarcophilusHarrisii*uniprot:G3V1L5 class I")
        == "Saha*uniprot:G3V1L5 class I"
    )
    assert (
        normalize_sequence_resolved_allele_name("TriakisScyllium-UAA*uniprot:O46817")
        == "TriakisScyllium-UAA*uniprot:O46817"
    )
    assert (
        normalize_sequence_resolved_allele_name("SarcophilusHarrisii*uniprot:G3V1L5 class I")
        == "Saha*uniprot:G3V1L5 class I"
    )
    assert allele_locus_name("HLA-A*uniprot:P04439") == "HLA-A"
    assert allele_locus_name("TriakisScyllium-UAA*uniprot:O46817") == "other"

    with pytest.raises(ValueError, match="MHC class II allele"):
        normalize_allele_name("HLA-DRA*uniprot:P01903")
    with pytest.raises(ValueError, match="MHC class II allele"):
        normalize_allele_name("Saha*uniprot:G3V1L5 class II")
    with pytest.raises(ValueError, match="MIC-family"):
        normalize_allele_name("HLA-MICA*uniprot:Q29983")
    with pytest.raises(ValueError, match="HFE-family"):
        normalize_allele_name("HLA-HFE*uniprot:Q30201")


def test_numpy_json_encoder_handles_numpy2_scalars_and_arrays():
    import json
    import numpy as np
    from mhcflurry.common import NumpyJSONEncoder

    payload = {
        "i64": np.int64(7),
        "u8": np.uint8(3),
        "f32": np.float32(1.5),
        "f64": np.float64(2.25),
        "flag": np.bool_(True),
        "arr": np.array([1, 2, 3]),
    }
    decoded = json.loads(json.dumps(payload, cls=NumpyJSONEncoder))
    assert decoded == {
        "i64": 7,
        "u8": 3,
        "f32": 1.5,
        "f64": 2.25,
        "flag": True,
        "arr": [1, 2, 3],
    }


def test_write_info_txt_format(tmp_path):
    from mhcflurry.model_provenance import write_info_txt

    info_path = write_info_txt(tmp_path, environ={})
    lines = open(info_path, encoding="utf-8").read().splitlines()
    keys = [line.split("\t", 1)[0] for line in lines]
    assert keys == ["trained on", "package   ", "hostname  ", "user      "]


def test_write_generate_sh_in_place_preserves_generate_sh(tmp_path):
    from mhcflurry.common import write_generate_sh

    orig = write_generate_sh(str(tmp_path), argv=["mhcflurry", "train"])
    cal = write_generate_sh(
        str(tmp_path),
        argv=["mhcflurry", "calibrate-percentile-ranks", "--models-dir", str(tmp_path)],
        filename="GENERATE_CALIBRATE.sh",
        in_place=True,
    )
    assert orig.endswith("GENERATE.sh")
    assert cal.endswith("GENERATE_CALIBRATE.sh")
    assert "train" in open(orig, encoding="utf-8").read()
    cal_text = open(cal, encoding="utf-8").read()
    assert "rm -rf" not in cal_text
    assert "calibrate-percentile-ranks" in cal_text
