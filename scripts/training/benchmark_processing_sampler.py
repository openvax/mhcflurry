#!/usr/bin/env python3
"""Reproducible CPU benchmark of processing candidate sampling (no model fits)."""

import argparse
import json
from pathlib import Path
import time

import numpy

from mhcflurry.proteome_decoys import sample_peptide_frame_for_accessions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proteins", type=int, default=1000)
    parser.add_argument("--protein-length", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.proteins < 1 or args.protein_length < 9 or args.sample_size < 1:
        parser.error("Require positive protein/sample counts and protein length >=9")
    if args.out.exists():
        raise ValueError("Use a fresh benchmark output to preserve previous evidence")
    rng = numpy.random.default_rng(args.seed)
    alphabet = numpy.array(list("ACDEFGHIKLMNPQRSTVWY"))
    sequences = {str(index): "".join(rng.choice(alphabet, args.protein_length))
                 for index in range(args.proteins)}
    records = []
    for method in ("reservoir", "positions"):
        numpy.random.seed(args.seed)
        started = time.perf_counter()
        frame = sample_peptide_frame_for_accessions(sequences, sequences, lengths=[8],
            n=args.sample_size, sampling_method=method)
        elapsed = time.perf_counter() - started
        record = {"method": method, "seconds": elapsed, "rows": len(frame)}
        records.append(record)
        print(record, flush=True)
    output = {"design": "synthetic-processing-sampler-benchmark-v1",
              "biological_performance_experiment": False,
              "parameters": {k: v for k, v in vars(args).items() if k != "out"},
              "numpy_version": numpy.__version__, "records": records,
              "speedup": records[0]["seconds"] / records[1]["seconds"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
