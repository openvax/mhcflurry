"""Preserve explicit training folds when replaying archived model inputs."""

import pandas


def extract_training_folds(data, num_folds, reuse=False):
    """Return data without fold columns and optionally validated stored folds."""
    # Remove malformed legacy merge suffixes as well when regenerating folds;
    # fail closed on them when the user requests exact reuse.
    existing = [name for name in data if name.startswith("fold_")]
    cleaned = data.drop(columns=existing)
    if not reuse:
        return cleaned, None
    expected = ["fold_%d" % fold for fold in range(num_folds)]
    if set(existing) != set(expected):
        raise ValueError("--reuse-folds requires exactly these columns: %s" % expected)
    folds = pandas.DataFrame(index=data.index)
    for name in expected:
        parsed = data[name].astype(str).str.lower().map({
            "true": True, "false": False, "1": True, "0": False})
        if parsed.isna().any():
            raise ValueError("Invalid or missing boolean values in %s" % name)
        if not parsed.any() or parsed.all():
            raise ValueError("%s must contain both training and validation rows" % name)
        folds[name] = parsed.astype(bool)
    return cleaned, folds
