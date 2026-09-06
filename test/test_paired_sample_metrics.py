import importlib.util
from pathlib import Path

import numpy
import pandas
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/training/paired_sample_metrics.py"
    spec = importlib.util.spec_from_file_location("paired_sample_metrics", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def example():
    return pandas.DataFrame({
        "sample": ["a", "b", "c"] * 2,
        "condition": ["old"] * 3 + ["new"] * 3,
        "metric": [0.1, 0.3, 0.4, 0.3, 0.5, 0.6], "n": [10] * 6})


def summarize(frame):
    return module().paired_summary(
        frame, units=["sample"], condition="condition", metrics=["metric"],
        baseline="old", replicates=100, seed=3)


def test_paired_samples_preserve_constant_difference_and_row_order_independence():
    summary, differences, draws = summarize(example().sample(frac=1, random_state=4))
    numpy.testing.assert_allclose(summary[["delta", "ci_low", "ci_high"]], 0.2)
    assert summary.samples_improved.item() == 3
    assert len(differences) == 3
    numpy.testing.assert_allclose(draws["new:metric"], 0.2)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite", "counts"])
def test_paired_samples_reject_invalid_comparisons(problem):
    frame = example()
    if problem == "missing":
        frame = frame.iloc[1:]
    elif problem == "duplicate":
        frame = pandas.concat([frame, frame.iloc[:1]])
    elif problem == "nonfinite":
        frame.loc[0, "metric"] = numpy.nan
    else:
        frame.loc[0, "n"] = 11
    with pytest.raises(ValueError):
        summarize(frame)
