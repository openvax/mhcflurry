#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write the paired hyperparameter panels from the 2.3.0 parity audit."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from mhcflurry.cli.generate_training_hyperparameters import (
    build_affinity_ablation_panels,
    build_processing_ablation_panels,
    build_processing_variant_grid,
)


def write_panels(out_dir):
    """Write ablation YAML files and a checksummed manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    def write_yaml(name, values):
        path = out_dir / name
        payload = yaml.safe_dump(values, sort_keys=True)
        path.write_text(payload)
        records.append({
            "path": name,
            "architectures": len(values),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        })

    for condition, values in build_affinity_ablation_panels().items():
        write_yaml("affinity.%s.yaml" % condition, values)

    for condition, base_values in build_processing_ablation_panels().items():
        for variant in ("with_flanks", "no_flank"):
            write_yaml(
                "processing.%s.%s.yaml" % (condition, variant),
                build_processing_variant_grid(base_values, variant),
            )

    manifest = {
        "schema_version": 1,
        "affinity_folds": 4,
        "processing_folds": 4,
        "records": records,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="Directory for YAML panels and manifest")
    args = parser.parse_args(argv)
    manifest = write_panels(args.out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
