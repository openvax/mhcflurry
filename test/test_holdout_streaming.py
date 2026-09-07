from types import SimpleNamespace

import pandas
import pytest

from mhcflurry.cli import compare_models


def write_benchmark(path, sample, hits=(1, 0)):
    pandas.DataFrame({"peptide": ["SIINFEKL", "SLYNTVATL"], "sample_id": [sample] * 2,
                      "hla": ["HLA-A*02:01"] * 2, "hit": hits,
                      "n_flank": ["AAA", "CCC"], "c_flank": ["GGG", "TTT"]}).to_csv(path, index=False)


@pytest.mark.parametrize("limit", [None, 1])
def test_streaming_preserves_holdout_rows_and_order(tmp_path, limit):
    data = tmp_path / "data"
    data.mkdir()
    for name, sample in (("a", "discard"), ("b", "keep"), ("c", "keep2")):
        write_benchmark(data / ("benchmark.multiallelic.train_excluded.%s.csv.bz2" % name), sample)
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    pandas.DataFrame({"sample_id": ["keep", "keep2"]}).to_csv(holdout / "presentation_samples.csv", index=False)
    args = SimpleNamespace(release_holdout_dir=str(holdout), limit_files=limit)
    original = compare_models._load_presentation_benchmark(str(data), None)
    expected = compare_models._filter_release_holdout_samples(original, args, "presentation").reset_index(drop=True)
    actual = compare_models._load_presentation_benchmark_for_component(str(data), args, "presentation")
    pandas.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize("bad_column,bad_value", [("hit", 0.5), ("peptide", ""), ("hla", None)])
def test_streaming_still_rejects_invalid_discarded_rows(tmp_path, bad_column, bad_value):
    path = tmp_path / "data.csv"
    write_benchmark(path, "discard")
    data = pandas.read_csv(path)
    data[bad_column] = data[bad_column].astype(object)
    data.loc[0, bad_column] = bad_value
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-binary|missing or blank"):
        compare_models._read_holdout_benchmark_files([str(path)], {"keep"}, "test")


def test_streaming_allows_single_class_chunks_and_keeps_file_order(tmp_path, monkeypatch):
    paths = [tmp_path / "positive.csv", tmp_path / "negative.csv"]
    for path, hits in zip(paths, [(1, 1), (0, 0)]):
        write_benchmark(path, "keep", hits)
    original_read = pandas.read_csv
    seen = []

    def read(*args, **kwargs):
        seen.append(kwargs.get("chunksize"))
        kwargs["chunksize"] = 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pandas, "read_csv", read)
    actual = compare_models._read_holdout_benchmark_files(list(map(str, paths)), {"keep"}, "test")
    assert seen == [100000, 100000]
    assert actual.hit.tolist() == [1, 1, 0, 0]
    assert actual.source_file.tolist() == ["positive.csv"] * 2 + ["negative.csv"] * 2


def test_streaming_affinity_matches_full_loader(tmp_path):
    for name, sample in (("a", "discard"), ("b", "keep")):
        write_benchmark(tmp_path / ("benchmark.monoallelic.mixmhcpred.train_excluded.%s.csv.bz2" % name), sample)
    all_rows = compare_models._load_affinity_benchmark(str(tmp_path), "mixmhcpred", None)
    actual = compare_models._load_affinity_benchmark(str(tmp_path), "mixmhcpred", None, samples={"keep"})
    pandas.testing.assert_frame_equal(all_rows.loc[all_rows.sample_id == "keep"].reset_index(drop=True), actual)
