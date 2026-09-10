"""Lossless processing CSV replay, including reference-score provenance."""

import io

import numpy
import pandas

from mhcflurry.training_folds import read_processing_training_data


def test_frozen_float_and_sample_identity_survive_repeated_roundtrips():
    text = "sample_id,affinity_prediction,fold_0\n001,182.06260894694452,True\n"
    expected = numpy.float64("182.06260894694452")
    for _ in range(5):
        frame = read_processing_training_data(io.StringIO(text))
        assert frame.sample_id.iloc[0] == "001"
        assert frame.affinity_prediction.iloc[0] == expected
        text = frame.to_csv(index=False)
    original = pandas.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    changed = text.replace("94694452", "9469445")
    observed = pandas.read_csv(io.StringIO(changed), dtype=str, keep_default_na=False)
    assert not original.equals(observed)


def test_numeric_only_fingerprint_is_not_inferred_as_an_integer():
    text = "sample_id,matching_affinity_reference_sha256\n001," + "0" * 64 + "\n"
    frame = read_processing_training_data(io.StringIO(text))
    assert frame.matching_affinity_reference_sha256.iloc[0] == "0" * 64
