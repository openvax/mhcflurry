#!/usr/bin/env python3
"""Expand and freeze unique 10:1 processing evaluation risk sets.

Original held-out hits and cached public affinities are retained. Only missing
negative capacity is supplied, from the same source-protein population and
frozen public predictor. All processing comparators consume the saved table.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy
import pandas

from mhcflurry.common import add_random_seed_arg, derive_seed, positive_int_arg
from mhcflurry.cli.compare_models import _load_presentation_benchmark_for_component
from mhcflurry.cli.processing_affinity_control import _attach_affinity, AFFINITY_COLUMN
from mhcflurry.experiment_archive import sha256_file
from mhcflurry.processing_matching import (
    MATCHING_POLICY, affinity_reference_fingerprint, validate_matched_training_data,
    validate_preparation_provenance)
from mhcflurry.processing_evaluation import hit_identity, verify_hits
from mhcflurry.processing_preparation import (
    SamplePreparation, add_processing_preparation_args, prepare_sample,
    write_csv_atomic, write_json_atomic)
from mhcflurry.proteome_decoys import (
    load_reference_sequences, sample_peptide_frame_for_accessions)


def expand_sample(args, pool, sequences, score):
    """Resume scored rounds, expanding only lengths with insufficient capacity."""
    sample = str(pool.sample_id.iloc[0])
    hits = pool.loc[pool.hit.eq(1)].copy()
    seed = derive_seed(args.random_seed, "processing-evaluation-expansion", sample)
    store = SamplePreparation(args, sample, hits)
    if store.load_pool() is None:
        store.save_round(pool, {"seed": seed, "origin": "frozen-benchmark-cache"})
    accessions = pool.protein_accession.dropna().drop_duplicates().tolist()
    flank_length = max(pool.n_flank.str.len().max(), pool.c_flank.str.len().max())
    genotype = pool.hla.iloc[0]

    def initial():
        raise AssertionError("The frozen benchmark pool must already be saved")

    def additional(length, excluded, round_seed):
        # The maintained string sampler uses numpy's RNG. This command is
        # serial, with a recorded, independent seed for each sample/round/length.
        numpy.random.seed(int(round_seed) % (2 ** 32))
        frame = sample_peptide_frame_for_accessions(
            accessions, sequences, lengths=[length], flanking_length=int(flank_length),
            exclude_peptides=excluded, n=args.expansion_candidates_per_length,
            allow_smaller=True).drop_duplicates("peptide")
        return frame.assign(sample_id=sample, hit=0, hla=genotype)

    result = prepare_sample(args, sample, hits, seed, initial, additional, score)
    verify_hits(pool, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--release-holdout-dir", required=True)
    parser.add_argument("--affinity-predictor", required=True)
    parser.add_argument("--proteome-reference-csv", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoys-per-hit", type=positive_int_arg, default=10)
    add_random_seed_arg(parser)
    add_processing_preparation_args(parser)
    args = parser.parse_args(argv)
    if args.resume_matching_dir:
        parser.error("Use --resume in this experiment's own output directory")
    args.limit_files = None
    destination = args.out
    args.out = destination / "matched.csv"
    args.max_affinity_distance = 0.25
    args.matching_reference = affinity_reference_fingerprint(args.affinity_predictor)
    cohort = _load_presentation_benchmark_for_component(args.data_dir, args, "processing")
    cohort, sources = _attach_affinity(cohort, args.data_dir)
    cohort["sample_id"] = cohort.sample_id.astype(str)
    cohort["affinity_prediction"] = cohort[AFFINITY_COLUMN]
    manifest = Path(args.release_holdout_dir) / "processing_samples.csv"
    inputs = {source["path"]: source["sha256"] for source in sources}
    inputs.update({str(Path(args.data_dir) / name): sha256_file(Path(args.data_dir) / name)
                   for name in cohort.source_file.unique()})
    inputs.update({str(path): sha256_file(path) for path in (manifest, Path(args.proteome_reference_csv))})
    provenance = {"policy": MATCHING_POLICY, "affinity_reference": args.matching_reference,
                  "decoys_per_hit": args.decoys_per_hit, "max_log10_affinity_distance": 0.25,
                  "candidate_pool_multiplier": None, "seed": args.random_seed, "inputs": inputs,
                  "expansion_candidates_per_length": args.expansion_candidates_per_length,
                  "preparation_script_sha256": sha256_file(__file__)}
    directory = Path(str(args.out) + ".matching")
    provenance_path = directory / "experiment.json"
    if destination.exists() and any(destination.iterdir()):
        if not args.resume:
            raise ValueError("Use a fresh output directory or --resume")
        saved = json.loads(provenance_path.read_text())
        validate_preparation_provenance(saved, provenance)
        if saved["expansion_candidates_per_length"] != args.expansion_candidates_per_length:
            raise ValueError("Changed expansion candidate count")
        if saved["preparation_script_sha256"] != provenance["preparation_script_sha256"]:
            raise ValueError("Changed preparation source; preserve the original experiment")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(provenance_path, provenance)
    sequences = load_reference_sequences(args.proteome_reference_csv, cohort.protein_accession)
    from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
    predictor = Class1PresentationPredictor(
        affinity_predictor=Class1AffinityPredictor.load(args.affinity_predictor))
    results = []
    checks = []
    for sample, pool in cohort.groupby("sample_id", sort=False):
        alleles = pool.hla.iloc[0].split()

        def score(peptides, genotype=alleles):
            return predictor.predict_affinity(
                peptides=list(peptides), alleles={"sample": genotype},
                include_affinity_percentile=False, verbose=0).affinity.to_numpy()

        # The historical cache lacks model hashes. Require the supplied
        # predictor to numerically reproduce a deterministic spread of its
        # scores before extending it, and record the exact check and tolerance.
        rows = numpy.unique(numpy.linspace(0, len(pool) - 1, min(256, len(pool)), dtype=int))
        expected = pool.affinity_prediction.to_numpy()[rows]
        actual = score(pool.peptide.iloc[rows])
        error = numpy.max(numpy.abs(numpy.log10(actual) - numpy.log10(expected)))
        if not numpy.isfinite(error) or error > 1e-4:
            raise ValueError("Affinity reference disagrees with frozen cache for %s: %g" % (sample, error))
        checks.append({"sample_id": sample, "rows": rows.tolist(), "max_log10_error": float(error)})
        results.append(expand_sample(args, pool.reset_index(drop=True), sequences, score))
    result = pandas.concat(results, ignore_index=True)
    verify_hits(cohort, result)
    validate_matched_training_data(result)
    output = destination / "matched.csv.bz2"
    write_csv_atomic(output, result)
    write_json_atomic(destination / "cohort.json", {
        **provenance, "rows": len(result), "hits": int(result.hit.sum()),
        "samples": int(result.sample_id.nunique()), "cohort_file": output.name,
        "cohort_sha256": sha256_file(output),
        "hit_identity_sha256": hashlib.sha256(hit_identity(result).tobytes()).hexdigest(),
        "reference_checks": checks, "reference_check_max_log10_tolerance": 1e-4,
        "purpose": "processing only; original presentation cohort unchanged"})
    print("Frozen processing cohort:", output, "rows:", len(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
