import importlib.util
from pathlib import Path

import pandas
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/training/processing_flank_ablation.py"
    spec = importlib.util.spec_from_file_location("processing_flank_ablation", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def example():
    return pandas.DataFrame({
        "source_row": [10, 11, 12, 13, 14, 15, 11],
        "sample_id": ["a"] * 4 + ["b"] * 2 + ["a"],
        "peptide": ["AAAAAAAA", "CCCCCCCC", "DDDDDDDD", "EEEEEEEEE",
                    "FFFFFFFF", "GGGGGGGG", "CCCCCCCC"],
        "hit": [1, 0, 0, 1, 1, 0, 0],
        "n_flank": ["AAAAA", "CCCCC", "DDDDD", "EEEEE", "FFFFF", "GGGGG", "CCCCC"],
        "c_flank": ["HHHHH", "IIIII", "KKKKK", "LLLLL", "MMMMM", "NNNNN", "IIIII"]})


def test_flank_shuffle_preserves_strata_pairs_and_duplicate_identity():
    result = module().perturbation_table(example(), 42)
    assert len(result) == 6
    donors = result.set_index("source_row").loc[result.donor_source_row].reset_index(drop=True)
    pandas.testing.assert_series_equal(result.sample_id, donors.sample_id)
    pandas.testing.assert_series_equal(result.peptide_len, donors.peptide_len)
    for flank in ("n_flank", "c_flank"):
        assert result["shuffled_" + flank].tolist() == donors[flank].tolist()
    again = module().perturbation_table(example().sample(frac=1, random_state=5), 42)
    pandas.testing.assert_frame_equal(result, again)


def test_flank_shuffle_does_not_condition_on_labels():
    original = module().perturbation_table(example(), 42)
    changed = example()
    changed["hit"] = 1 - changed.hit
    result = module().perturbation_table(changed, 42)
    pandas.testing.assert_series_equal(original.donor_source_row, result.donor_source_row)


@pytest.mark.parametrize("column,value", [("peptide", "YYYYYYYY"), ("hit", 1), ("n_flank", "VVVVV")])
def test_flank_shuffle_rejects_conflicting_repeated_rows(column, value):
    frame = example()
    frame.loc[6, column] = value
    with pytest.raises(ValueError, match="inconsistent"):
        module().perturbation_table(frame, 42)


def test_flank_audit_uses_local_context_and_counts_unknowns():
    frame = example().iloc[:1].copy()
    frame["n_flank"] = "XXXXXAAAAA"
    frame["c_flank"] = "AAAAXAAAAA"
    audit = module().flank_audit(frame).iloc[0]
    assert audit.n_complete_fraction == 1
    assert audit.c_complete_fraction == 0
    frame["n_flank"] = None
    assert module().flank_audit(frame).iloc[0].n_empty_fraction == 1
