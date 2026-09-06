import importlib.util
from pathlib import Path

import pandas


def test_identity_preserves_duplicates_and_reports_order_separately():
    path = Path(__file__).resolve().parents[1] / "scripts/training/audit_training_data_identity.py"
    spec = importlib.util.spec_from_file_location("training_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = pandas.DataFrame({"peptide": ["A", "A", "B"], "hit": [1, 1, 0]})
    same = old.iloc[::-1]
    summary, differences = module.compare_rows(old, same, ["peptide", "hit"])
    assert summary["same_row_multiset"]
    assert not summary["same_ordered_rows"]
    assert differences.empty
    new = pandas.DataFrame({"peptide": ["A", "B", "B"], "hit": [1, 0, 1]})
    summary, differences = module.compare_rows(old, new, ["peptide", "hit"])
    assert not summary["same_row_multiset"]
    assert summary["shared_rows_with_multiplicity"] == 2
    assert summary["reference_only_rows_with_multiplicity"] == 1
    assert summary["candidate_only_rows_with_multiplicity"] == 1
    assert len(differences) == 2
