"""Unit tests for ``mhcflurry.select_pan_allele_models_command``."""
import inspect


def test_fold_col_parser_rejects_pandas_merge_suffixes():
    """Regression: when train_data has been pandas-merged with stale
    fold_* columns still attached, the saved metadata DataFrame can carry
    ``fold_0_x``/``fold_0_y`` etc. The select fold-col parser must
    surface that as a clean error rather than crashing later in
    ``int(col.split("_")[-1])``."""
    from mhcflurry import select_pan_allele_models_command as mod

    src = inspect.getsource(mod.run)
    assert r'r"^fold_\d+$"' in src or r"r'^fold_\d+$'" in src, (
        "fold col parser must use ^fold_<int>$ regex; otherwise "
        "fold_0_x slips through and crashes int() later"
    )
