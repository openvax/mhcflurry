"""Unit tests for ``mhcflurry.select_processing_models_command``."""
import inspect


def test_fold_col_parser_rejects_pandas_merge_suffixes():
    """Regression: ``fold_0_x`` must not be parsed as a real fold column."""
    from mhcflurry import select_processing_models_command as mod

    src = inspect.getsource(mod.run)
    assert r'r"^fold_\d+$"' in src or r"r'^fold_\d+$'" in src, (
        "fold col parser must use ^fold_<int>$ regex; otherwise "
        "fold_0_x slips through and crashes int() later"
    )
