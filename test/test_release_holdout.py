# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
from types import SimpleNamespace

import pandas
import pytest

from mhcflurry.cli.reassign_mass_spec_training_data import (
    reassign_mass_spec_training_data,
)
from mhcflurry.cli.compare_models import (
    _filter_release_holdout_samples,
    _load_presentation_benchmark_for_component,
)
from mhcflurry.release_holdout import (
    AFFINITY_PMHCS_FILE,
    AFFINITY_SAMPLES_FILE,
    PRESENTATION_SAMPLES_FILE,
    PROCESSING_SAMPLES_FILE,
    build_release_holdout,
    exclude_samples,
    validate_release_holdout,
)


def _write_benchmark(path, rows):
    pandas.DataFrame(rows).to_csv(path, index=False)


def test_build_apply_and_validate_release_holdout(tmp_path, monkeypatch):
    data_dir = tmp_path / "data_evaluation"
    data_dir.mkdir()
    _write_benchmark(
        data_dir / "benchmark.monoallelic.train_excluded.csv.bz2",
        [
            {
                "peptide": "SIINFEKL",
                "sample_id": "mono-test",
                "hit": 1,
                "hla": "HLA-A*02:01",
            },
            {
                "peptide": "AAAAAAAAA",
                "sample_id": "mono-test",
                "hit": 0,
                "hla": "HLA-A*02:01",
            },
        ],
    )
    _write_benchmark(
        data_dir / "benchmark.multiallelic.train_excluded.csv.bz2",
        [
            {
                "peptide": "GILGFVFTL",
                "sample_id": "multi-test",
                "hit": 1,
                "hla": "HLA-A*02:01 HLA-B*07:02",
            },
        ],
    )
    affinity_input = tmp_path / "affinity.csv"
    pandas.DataFrame([
        {
            "allele": "HLA-A0201",
            "peptide": "SIINFEKL",
            "measurement_value": 100.0,
            "measurement_inequality": "<",
            "measurement_type": "qualitative",
            "measurement_kind": "mass_spec",
        },
        {
            "allele": "HLA-A*02:01",
            "peptide": "AAAAAAAAA",
            "measurement_value": 50000.0,
            "measurement_inequality": ">",
            "measurement_type": "qualitative",
            "measurement_kind": "mass_spec",
        },
        {
            "allele": "HLA-B*07:02",
            "peptide": "GILGFVFTL",
            "measurement_value": 100.0,
            "measurement_inequality": "<",
            "measurement_type": "qualitative",
            "measurement_kind": "mass_spec",
        },
        {
            "allele": "HLA-A*03:01",
            "peptide": "KLGGALQAK",
            "measurement_value": 50.0,
            "measurement_inequality": "=",
            "measurement_type": "quantitative",
            "measurement_kind": "affinity",
        },
    ]).to_csv(affinity_input, index=False)
    mass_spec_input = tmp_path / "annotated_ms.csv"
    pandas.DataFrame([
        {
            "sample_id": "mono-test",
            "pmid": "31844290",
            "format": "MONOALLELIC",
            "mhc_class": "I",
        },
        {
            "sample_id": "multi-test",
            "pmid": "31154438",
            "format": "MULTIALLELIC",
            "mhc_class": "I",
        },
    ]).to_csv(mass_spec_input, index=False)
    holdout_dir = tmp_path / "holdout"
    policy = build_release_holdout(
        data_dir,
        affinity_input,
        mass_spec_input,
        holdout_dir,
        chunksize=1,
    )

    assert policy["evaluation_hit_counts"] == {
        "monoallelic": 1,
        "multiallelic": 1,
    }
    assert policy["affinity_pmhc_count"] == 3
    assert set(map(tuple, pandas.read_csv(
        holdout_dir / AFFINITY_PMHCS_FILE).to_numpy())) == {
            ("HLA-A*02:01", "SIINFEKL"),
            ("HLA-A*02:01", "AAAAAAAAA"),
            ("HLA-B*07:02", "GILGFVFTL"),
        }
    assert pandas.read_csv(
        holdout_dir / AFFINITY_SAMPLES_FILE).sample_id.tolist() == [
            "mono-test",
        ]

    affinity_output = tmp_path / "affinity.filtered.csv"
    filtered = reassign_mass_spec_training_data(
        affinity_input,
        exclude_pmhcs=holdout_dir / AFFINITY_PMHCS_FILE,
        out_csv=affinity_output,
    )
    assert list(map(tuple, filtered[["allele", "peptide"]].to_numpy())) == [
        ("HLA-A*03:01", "KLGGALQAK"),
    ]

    processing = pandas.DataFrame([
        {"sample_id": "multi-test", "peptide": "SIINFEKL", "hit": 1},
        {"sample_id": "mono-train", "peptide": "KLGGALQAK", "hit": 1},
    ])
    processing = exclude_samples(
        processing,
        holdout_dir / PROCESSING_SAMPLES_FILE,
        "processing",
    )
    processing_path = tmp_path / "processing.csv"
    processing.to_csv(processing_path, index=False)

    presentation = pandas.DataFrame([
        {"sample_id": "multi-test", "peptide": "GILGFVFTL", "hit": 1},
        {"sample_id": "multi-train", "peptide": "KLGGALQAK", "hit": 1},
        # Production presentation data mixes numeric-looking and textual IDs.
        {"sample_id": "24616531", "peptide": "AAAAAAAAA", "hit": 0},
    ])
    presentation = exclude_samples(
        presentation,
        holdout_dir / PRESENTATION_SAMPLES_FILE,
        "presentation",
    )
    presentation_path = tmp_path / "presentation.csv"
    presentation.to_csv(presentation_path, index=False)

    sample_read_options = []
    read_csv = pandas.read_csv

    def read_csv_spy(path, *args, **kwargs):
        if (
                path in (processing_path, presentation_path)
                and kwargs.get("usecols") == ["sample_id"]):
            sample_read_options.append(kwargs.copy())
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pandas, "read_csv", read_csv_spy)

    validation_path = tmp_path / "validation.json"
    result = validate_release_holdout(
        holdout_dir,
        affinity_output,
        processing_path,
        presentation_path,
        out=validation_path,
    )
    assert len(sample_read_options) == 2
    assert all(
        options["dtype"] == {"sample_id": str}
        for options in sample_read_options
    )
    assert all(options["chunksize"] for options in sample_read_options)
    assert result["schema_version"] == 1
    assert result["affinity_overlap_rows"] == 0
    assert result["processing_overlap_rows"] == 0
    assert result["presentation_overlap_rows"] == 0
    assert result["policy_sha256"]
    assert set(result["holdout_files"]) == {
        "affinity_pmhcs.csv",
        "affinity_samples.csv",
        "processing_samples.csv",
        "presentation_samples.csv",
    }
    assert json.loads(validation_path.read_text()) == result

    processing.loc[len(processing)] = ["multi-test", "SIINFEKL", 1]
    processing.to_csv(processing_path, index=False)
    with pytest.raises(ValueError, match="processing_overlap_rows.*1"):
        validate_release_holdout(
            holdout_dir,
            affinity_output,
            processing_path,
            presentation_path,
        )

    benchmark = pandas.DataFrame({
        "sample_id": ["mono-test", "multi-test", "multi-train"],
        "peptide": ["SIINFEKL", "GILGFVFTL", "KLGGALQAK"],
    })
    args = SimpleNamespace(
        release_holdout_dir=str(holdout_dir),
        limit_files=None,
    )
    affinity_eval = _filter_release_holdout_samples(
        benchmark, args, "affinity")
    processing_eval = _filter_release_holdout_samples(
        benchmark, args, "processing")
    assert affinity_eval.sample_id.tolist() == ["mono-test"]
    assert processing_eval.sample_id.tolist() == ["multi-test"]


def test_release_holdout_file_limit_applies_before_genotype_normalization(
        tmp_path, monkeypatch):
    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    pandas.DataFrame({
        "sample_id": ["held-out"],
    }).to_csv(holdout_dir / PROCESSING_SAMPLES_FILE, index=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for filename, sample_id, peptides, genotype in (
            (
                "a",
                "training",
                ["AAAAAAAA", "AAAAAAAK"],
                "excluded genotype must not be normalized",
            ),
            (
                "z",
                "held-out",
                ["BBBBBBBB", "BBBBBBBK"],
                "HLA-A*02:01",
            )):
        pandas.DataFrame({
            "sample_id": [sample_id, sample_id],
            "peptide": peptides,
            "hla": [genotype, genotype],
            "hit": [1, 0],
        }).to_csv(
            data_dir / (
                "benchmark.multiallelic.train_excluded.%s.csv.bz2" % filename
            ),
            index=False,
        )
    args = SimpleNamespace(
        release_holdout_dir=str(holdout_dir),
        limit_files=1,
    )
    normalized = []

    def normalize_selected_genotype(value):
        normalized.append(value)
        if value != "HLA-A*02:01":
            raise AssertionError("normalized a non-holdout genotype")
        return value

    monkeypatch.setattr(
        "mhcflurry.cli.compare_models._normalize_benchmark_genotype",
        normalize_selected_genotype,
    )

    result = _load_presentation_benchmark_for_component(
        str(data_dir),
        args,
        "processing",
    )

    assert result.sample_id.tolist() == ["held-out", "held-out"]
    assert result.source_file.unique().tolist() == [
        "benchmark.multiallelic.train_excluded.z.csv.bz2",
    ]
    assert normalized == ["HLA-A*02:01", "HLA-A*02:01"]
