#!/usr/bin/env python3
"""Make an affinity checkpoint self-contained with its pseudosequences."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from mhcflurry import Class1AffinityPredictor


def sha256_file(path):
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fd:
        for block in iter(lambda: fd.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_directory(path):
    """Hash all files in a directory by relative path and content."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        item_hash = sha256_file(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\0")
        files.append({"path": relative, "sha256": item_hash})
    return {"sha256": digest.hexdigest(), "files": files}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--allele-sequences", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def run(args):
    source = Path(args.models).resolve()
    allele_sequences = Path(args.allele_sequences).resolve()
    out = Path(args.out).resolve()
    if not (source / "manifest.csv").is_file():
        raise ValueError("Missing affinity manifest: %s" % source)
    if allele_sequences.name not in (
        "allele_sequences.csv",
        "allele_sequences.csv.bz2",
    ):
        raise ValueError("Unexpected pseudosequence filename: %s" % allele_sequences)
    if out.exists():
        raise ValueError("Output already exists: %s" % out)

    shutil.copytree(source, out)
    copied_sequences = out / allele_sequences.name
    shutil.copy2(allele_sequences, copied_sequences)
    predictor = Class1AffinityPredictor.load(str(out), optimization_level=0)
    provenance = {
        "format": 1,
        "source_models": {
            "path": str(source),
            "fingerprint": fingerprint_directory(source),
        },
        "allele_sequences": {
            "path": str(allele_sequences),
            "sha256": sha256_file(allele_sequences),
        },
        "assembled_predictor": {
            "path": str(out),
            "models": int(len(predictor.neural_networks)),
            "supported_alleles": int(len(predictor.supported_alleles)),
        },
    }
    with open(out / "assembly_provenance.json", "w") as fd:
        json.dump(provenance, fd, indent=2, sort_keys=True)
        fd.write("\n")
    print(json.dumps(provenance["assembled_predictor"], sort_keys=True))


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
