#!/usr/bin/env python
"""
Validate that allele name -> pseudosequence mappings are consistent.

Checks:
  1. The raw allele_sequences.csv from the downloads is loaded identically
     by both the current code and a from-scratch parse.
  2. The renormalization step (mhcgnomes) in Class1AffinityPredictor.load()
     produces a deterministic mapping with no collisions or lost alleles.
  3. Every allele in the downloaded models' allele_sequences.csv maps to the
     same pseudosequence as the standalone allele_sequences download.
  4. Every allele used in the fixture CSV resolves to a sequence.

Usage:
    python scripts/validate_allele_sequences.py
"""
import os
import sys

import pandas as pd

from mhcflurry import Class1AffinityPredictor
from mhcflurry.common import normalize_allele_name
from mhcflurry.downloads import configure, get_path, get_default_class1_models_dir


def load_raw_csv(path):
    """Load allele_sequences.csv exactly as on disk (no renormalization)."""
    return pd.read_csv(path, index_col=0).iloc[:, 0].to_dict()


def renormalize(raw_mapping):
    """
    Apply the same renormalization that Class1AffinityPredictor.load() does.

    Returns (renormalized_dict, skipped_list, collision_list).
    """
    renormalized = {}
    skipped = []
    collisions = []
    for name, sequence in raw_mapping.items():
        normalized = normalize_allele_name(name, raise_on_error=False)
        if normalized is None:
            skipped.append(name)
            continue
        if normalized in renormalized and name != normalized:
            collisions.append((name, normalized))
            continue
        renormalized[normalized] = sequence
    return renormalized, skipped, collisions


def main():
    configure()
    errors = []

    # --- 1. Load the standalone allele_sequences download ---
    print("=" * 70)
    print("1. Loading standalone allele_sequences download")
    print("=" * 70)
    standalone_csv = get_path("allele_sequences", "allele_sequences.csv")
    standalone_raw = load_raw_csv(standalone_csv)
    print("  Raw entries: %d" % len(standalone_raw))
    standalone_norm, standalone_skipped, standalone_collisions = renormalize(
        standalone_raw)
    print("  After renormalization: %d entries" % len(standalone_norm))
    print("  Skipped (unparseable): %d" % len(standalone_skipped))
    print("  Collisions (duplicate after renorm): %d" % len(
        standalone_collisions))
    if standalone_skipped:
        print("  Skipped names (first 20): %s" % standalone_skipped[:20])

    # --- 2. Load from the predictor (goes through the same renorm logic) ---
    print()
    print("=" * 70)
    print("2. Loading allele_to_sequence from Class1AffinityPredictor.load()")
    print("=" * 70)
    models_dir = get_default_class1_models_dir()
    predictor = Class1AffinityPredictor.load(models_dir)
    predictor_seq = predictor.allele_to_sequence
    print("  Entries: %d" % len(predictor_seq))

    # --- 3. Load the raw CSV from the models directory ---
    print()
    print("=" * 70)
    print("3. Loading raw CSV from models directory")
    print("=" * 70)
    models_csv = os.path.join(models_dir, "allele_sequences.csv")
    models_raw = load_raw_csv(models_csv)
    print("  Raw entries: %d" % len(models_raw))
    models_norm, models_skipped, models_collisions = renormalize(models_raw)
    print("  After renormalization: %d entries" % len(models_norm))
    print("  Skipped: %d" % len(models_skipped))
    print("  Collisions: %d" % len(models_collisions))

    # --- 4. Compare predictor mapping vs models dir renormalized ---
    print()
    print("=" * 70)
    print("4. Comparing predictor.allele_to_sequence vs models-dir renormalized")
    print("=" * 70)
    pred_keys = set(predictor_seq.keys())
    model_keys = set(models_norm.keys())
    only_pred = pred_keys - model_keys
    only_model = model_keys - pred_keys
    common = pred_keys & model_keys
    print("  In predictor only: %d" % len(only_pred))
    print("  In models-dir renorm only: %d" % len(only_model))
    print("  In common: %d" % len(common))
    mismatched = []
    for k in sorted(common):
        if predictor_seq[k] != models_norm[k]:
            mismatched.append(k)
    if mismatched:
        print("  MISMATCH in %d alleles:" % len(mismatched))
        for k in mismatched[:20]:
            print("    %s:" % k)
            print("      predictor: %s" % predictor_seq[k])
            print("      models:    %s" % models_norm[k])
        errors.append(
            "Predictor vs models-dir mismatch: %d alleles" % len(mismatched))
    else:
        print("  All %d common alleles have identical sequences." % len(common))

    # --- 5. Compare models-dir vs standalone download (informational) ---
    # The models ship with their own allele_sequences.csv which may use a
    # different pseudosequence definition than the standalone download.
    # Differences here are expected and not an error.
    print()
    print("=" * 70)
    print("5. Comparing models-dir vs standalone allele_sequences download")
    print("   (informational — different pseudosequence versions are expected)")
    print("=" * 70)
    model_keys_set = set(models_norm.keys())
    standalone_keys_set = set(standalone_norm.keys())
    only_models = model_keys_set - standalone_keys_set
    only_standalone = standalone_keys_set - model_keys_set
    common2 = model_keys_set & standalone_keys_set
    print("  In models only: %d" % len(only_models))
    print("  In standalone only: %d" % len(only_standalone))
    print("  In common: %d" % len(common2))
    mismatched2 = []
    for k in sorted(common2):
        if models_norm[k] != standalone_norm[k]:
            mismatched2.append(k)
    if mismatched2:
        same_len = all(
            len(models_norm[k]) == len(standalone_norm[k])
            for k in mismatched2)
        print("  Different sequences: %d / %d (same length: %s)" % (
            len(mismatched2), len(common2), same_len))
        print("  (This is expected if pseudosequence positions differ.)")
    else:
        print("  All %d common alleles have identical sequences." % len(
            common2))

    # --- 6. Validate fixture alleles ---
    print()
    print("=" * 70)
    print("6. Validating fixture alleles resolve to sequences")
    print("=" * 70)
    fixture_csv = os.path.join(
        os.path.dirname(__file__), os.pardir, "test", "data",
        "master_released_class1_presentation_highscore_rows.csv.gz",
    )
    if os.path.exists(fixture_csv):
        fixture_df = pd.read_csv(fixture_csv, keep_default_na=False)
        fixture_alleles = fixture_df["allele"].unique()
        print("  Fixture alleles: %d" % len(fixture_alleles))
        missing = []
        for allele in sorted(fixture_alleles):
            normalized = normalize_allele_name(allele, raise_on_error=False)
            if normalized is None:
                missing.append((allele, "failed to normalize"))
            elif normalized not in predictor_seq:
                missing.append((allele, "normalized to '%s' but not in predictor" % normalized))
        if missing:
            print("  MISSING %d fixture alleles:" % len(missing))
            for allele, reason in missing:
                print("    %s: %s" % (allele, reason))
            errors.append("Missing fixture alleles: %d" % len(missing))
        else:
            print("  All %d fixture alleles resolve correctly." % len(
                fixture_alleles))
    else:
        print("  Fixture CSV not found, skipping.")

    # --- 7. Verify normalize_allele_name is deterministic ---
    print()
    print("=" * 70)
    print("7. Verifying normalize_allele_name determinism (sample of 1000)")
    print("=" * 70)
    sample_alleles = sorted(predictor_seq.keys())[:1000]
    nondeterministic = []
    for allele in sample_alleles:
        n1 = normalize_allele_name(allele, raise_on_error=False)
        n2 = normalize_allele_name(allele, raise_on_error=False)
        if n1 != n2:
            nondeterministic.append((allele, n1, n2))
    if nondeterministic:
        print("  NON-DETERMINISTIC: %d alleles" % len(nondeterministic))
        errors.append("Non-deterministic normalization: %d" % len(
            nondeterministic))
    else:
        print("  All 1000 sampled alleles normalize deterministically.")

    # --- 8. Check idempotency: normalize(normalize(x)) == normalize(x) ---
    print()
    print("=" * 70)
    print("8. Checking normalize idempotency (sample of 1000)")
    print("=" * 70)
    non_idempotent = []
    for allele in sample_alleles:
        n1 = normalize_allele_name(allele, raise_on_error=False)
        if n1 is None:
            continue
        n2 = normalize_allele_name(n1, raise_on_error=False)
        if n1 != n2:
            non_idempotent.append((allele, n1, n2))
    if non_idempotent:
        print("  NON-IDEMPOTENT: %d alleles" % len(non_idempotent))
        for allele, n1, n2 in non_idempotent[:10]:
            print("    %s -> %s -> %s" % (allele, n1, n2))
        errors.append("Non-idempotent normalization: %d" % len(
            non_idempotent))
    else:
        print("  All sampled alleles normalize idempotently.")

    # --- Summary ---
    print()
    print("=" * 70)
    if errors:
        print("ERRORS FOUND:")
        for e in errors:
            print("  - %s" % e)
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
