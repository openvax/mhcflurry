"""Reproducible sampler, matching and optional real-predictor parity timings."""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import time

import numpy
import pandas
import torch

from mhcflurry.amino_acid import COMMON_AMINO_ACIDS
from mhcflurry.common import positive_int_arg
from mhcflurry.experiment_archive import sha256_file
from mhcflurry.numeric_proteome import ProcessingProteome
from mhcflurry.processing_matching import matched_training_data
from mhcflurry.proteome_decoys import sample_peptide_frame_for_accessions


def timed(call, repeats):
    times = []
    cpu_times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        cpu_start = time.process_time()
        result = call()
        times.append(time.perf_counter() - start)
        cpu_times.append(time.process_time() - cpu_start)
    return result, {"seconds": times, "median_seconds": statistics.median(times),
                    "cpu_seconds": cpu_times, "median_cpu_seconds": statistics.median(cpu_times)}


def timed_pair(first, second, repeats):
    """Alternate comparison order to avoid separate-block load/warmup bias."""
    results = [None, None]
    timings = [{"seconds": [], "cpu_seconds": []} for _ in range(2)]
    orders = []
    for repeat in range(repeats):
        order = [0, 1] if repeat % 2 == 0 else [1, 0]
        orders.append(order)
        for index in order:
            results[index], measured = timed((first, second)[index], 1)
            for key in ("seconds", "cpu_seconds"):
                timings[index][key].extend(measured[key])
    for measured in timings:
        measured["median_seconds"] = statistics.median(measured["seconds"])
        measured["median_cpu_seconds"] = statistics.median(measured["cpu_seconds"])
    return results, timings, orders


def run(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--proteins", type=positive_int_arg, default=100)
    parser.add_argument("--protein-length", type=positive_int_arg, default=1000)
    parser.add_argument("--draws", type=positive_int_arg, default=25000)
    parser.add_argument("--repeats", type=positive_int_arg, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scored-pool", action="append", default=[],
                        help="Saved real scored pool(s) to concatenate for matching parity.")
    parser.add_argument("--baseline-matching-source",
                        help="Frozen historical processing_matching.py for exact baseline replay.")
    parser.add_argument("--affinity-predictor", help="Optional actual weights for numeric/string prediction parity.")
    parser.add_argument("--allele", default="HLA-A*02:01")
    parsed = parser.parse_args(args)
    out = Path(parsed.out)
    if out.exists():
        raise FileExistsError(out)
    rng = numpy.random.RandomState(parsed.seed)
    alphabet = numpy.asarray(sorted(COMMON_AMINO_ACIDS))
    proteins = {"p%d" % i: "".join(rng.choice(alphabet, parsed.protein_length)) for i in range(parsed.proteins)}
    start = time.perf_counter()
    index = ProcessingProteome(proteins)
    setup = time.perf_counter() - start

    def original():
        numpy.random.seed(parsed.seed)
        return sample_peptide_frame_for_accessions(proteins, proteins, [8], 5,
            n=parsed.draws, allow_smaller=True).drop_duplicates("peptide").reset_index(drop=True)

    baseline, baseline_timing = timed(original, parsed.repeats)
    windows, numeric_timing = timed(lambda: index.sample(proteins, 8, parsed.draws, (),
        numpy.random.RandomState(parsed.seed)), parsed.repeats)
    exported, export_timing = timed(lambda: windows.to_frame(5), parsed.repeats)
    pandas.testing.assert_frame_equal(exported, baseline)
    result = {
        "parameters": vars(parsed), "python": platform.python_version(),
        "numpy": numpy.__version__, "pandas": pandas.__version__, "torch": torch.__version__,
        "sampler": {"dataset": "synthetic", "protein_encoding_seconds": setup,
            "previous_numeric_positions_with_strings": baseline_timing,
            "reusable_numeric_positions_without_strings": numeric_timing,
            "export_strings_seconds": export_timing, "exact_export_parity": True},
        "source_hashes": {str(path): sha256_file(path) for path in [Path(__file__), *[
            Path(__file__).resolve().parents[2] / relative for relative in (
                "mhcflurry/numeric_proteome.py", "mhcflurry/numeric_sequences.py",
                "mhcflurry/processing_matching.py", "mhcflurry/scoring_pipeline.py",
                "mhcflurry/processing_preparation.py", "mhcflurry/class1_encoding.py",
                "mhcflurry/class1_neural_network.py", "mhcflurry/class1_affinity_predictor.py",
                "scripts/training/release_exact/make_train_data.processing.py")]]}}
    if parsed.scored_pool:
        frames = [pandas.read_csv(path, low_memory=False) for path in parsed.scored_pool]
        frame = pandas.concat(frames, ignore_index=True)
        reference = {"sha256": "benchmark-only"}
        def compare():
            return matched_training_data(frame, reference)
        result["matching"] = {"inputs": {p: sha256_file(p) for p in parsed.scored_pool},
            "rows": len(frame), "missing_protein_rows": int(frame.protein_accession.isna().sum())}
        if parsed.baseline_matching_source:
            spec = importlib.util.spec_from_file_location("mhcflurry._benchmark_baseline", parsed.baseline_matching_source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            (actual, expected), (timing, original_timing), orders = timed_pair(compare,
                lambda: module.matched_training_data(frame, reference), parsed.repeats)
            pandas.testing.assert_frame_equal(actual[0], expected[0])
            assert actual[1] == expected[1]
            result["matching"].update(baseline=original_timing, exact_assignment_and_diagnostics_parity=True,
                baseline_source_sha256=sha256_file(parsed.baseline_matching_source),
                execution_order=orders, execution_order_labels=["vectorized", "baseline"],
                speedup=original_timing["median_seconds"] / timing["median_seconds"])
        else:
            actual, timing = timed(compare, parsed.repeats)
        result["matching"].update(matched_rows=len(actual[0]), vectorized=timing)
    if parsed.affinity_predictor:
        from mhcflurry import Class1AffinityPredictor
        from mhcflurry.processing_matching import affinity_reference_fingerprint
        predictor = Class1AffinityPredictor.load(parsed.affinity_predictor)
        allele = predictor.canonicalize_allele_name(parsed.allele)
        models = predictor.class1_pan_allele_models or predictor.allele_to_allele_specific_models[allele]
        device = models[0].get_device()
        numeric = index.gather(windows, device)
        # Warm both paths before timing. Returned NumPy predictions synchronize
        # accelerator work, so timings include actual completed inference.
        predictor.predict(exported.peptide, allele=allele)
        predictor.predict_numeric(numeric, allele=allele)
        expected, old_time = timed(lambda: predictor.predict(exported.peptide, allele=allele), parsed.repeats)
        actual, new_time = timed(lambda: predictor.predict_numeric(numeric, allele=allele), parsed.repeats)
        numpy.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        result["prediction"] = {"device": str(device), "rows": len(numeric),
            "reference": affinity_reference_fingerprint(parsed.affinity_predictor),
            "string": old_time, "numeric": new_time, "max_absolute_difference": float(numpy.max(numpy.abs(actual - expected))),
            "allclose_rtol": 1e-6, "allclose_atol": 1e-6}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
