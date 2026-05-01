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


def test_train_peptide_hash_computed_before_allele_filter():
    """Regression: the per-fold train_peptide_hash must be computed on
    the raw saved ``train_data.csv.bz2`` *before* the canonicalizable-
    allele filter is applied. The saved file is exactly the data
    ``train_pan_allele_models_command`` trained on; if select hashes
    a strictly narrower subset (because ``filter_canonicalizable_alleles``
    drops a pseudogene/null annotation that train kept), the
    ``numpy.testing.assert_equal`` against ``training_info['train_peptide_hash']``
    fires a false-positive and aborts model selection.

    Verifies (via source inspection) that the hash is built over
    ``df.loc[df[col] == 1]`` while ``df`` is still the raw read, i.e.
    before the ``df = df.loc[df.allele.isin(alleles)]`` reassignment."""
    from mhcflurry import select_pan_allele_models_command as mod

    src = inspect.getsource(mod.run)
    hash_idx = src.index("make_train_peptide_hash(df.loc[df[col] == 1])")
    allele_filter_idx = src.index("df = df.loc[df.allele.isin(alleles)]")
    assert hash_idx < allele_filter_idx, (
        "train_peptide_hash must be computed before the canonicalizable-"
        "allele filter narrows df; otherwise hashes mismatch when any "
        "training allele is non-canonicalizable"
    )
