#!/usr/bin/env python3
"""Compare saved processing predictions with a reference boundary extractor."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy
import pandas
import torch

from mhcflurry import Class1ProcessingPredictor
from mhcflurry.class1_processing_neural_network import Class1ProcessingModel
from mhcflurry.cli.processing_affinity_control import score_risk_sets
from mhcflurry.common import configure_pytorch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    filename = "mhcflurry/class1_processing_neural_network.py"
    source = subprocess.check_output(
        ["git", "show", "%s:%s" % (args.reference_commit, filename)],
        cwd=repo, text=True)
    model_class = next(node for node in ast.parse(source).body
                       if isinstance(node, ast.ClassDef) and node.name == "Class1ProcessingModel")
    method = next(node for node in model_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_extract_boundary_windows")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
    configure_pytorch(backend="cpu", num_threads=2)
    frame = pandas.read_csv(args.predictions, dtype={"sample_id": str}, low_memory=False)
    predictor = Class1ProcessingPredictor.load(args.models)
    if any(model.hyperparameters.get("cleavage_boundary_peptide_length", 0)
           > frame.peptide.str.len().min() for model in predictor.models):
        raise ValueError("Reference extractor requires peptide context within every peptide")
    kwargs = dict(peptides=frame.peptide, n_flanks=frame.n_flank, c_flanks=frame.c_flank)
    frame["candidate_score"] = predictor.predict(**kwargs)
    with patch.object(Class1ProcessingModel, "_extract_boundary_windows",
                      namespace["_extract_boundary_windows"]):
        frame["reference_score"] = predictor.predict(**kwargs)
    delta = frame.candidate_score.to_numpy() - frame.reference_score.to_numpy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "matched_predictions.csv.bz2", index=False)
    metrics = score_risk_sets(frame, ["candidate_score", "reference_score"])
    metrics.to_csv(out / "metrics.csv", index=False)
    result = {
        "arguments": vars(args), "rows": len(frame), "models": len(predictor.models),
        "max_absolute_difference": float(numpy.abs(delta).max()),
        "exactly_equal": bool(numpy.array_equal(frame.candidate_score, frame.reference_score)),
        "reference_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256((repo / filename).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256((Path(args.models) / "manifest.csv").read_bytes()).hexdigest(),
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["exactly_equal"]:
        raise SystemExit("Boundary compatibility check changed predictions")


if __name__ == "__main__":
    main()
