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

"""Materialize retained affinity checkpoints as an ordinary predictor."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pandas

from ..class1_affinity_predictor import Class1AffinityPredictor


POLICIES = ("terminal", "best")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_parser(prog="mhcflurry train materialize-affinity-checkpoint"):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--models-dir",
        required=True,
        help="Predictor directory containing retained checkpoint sidecars.",
    )
    parser.add_argument(
        "--policy",
        required=True,
        choices=POLICIES,
        help="Checkpoint policy to install as the predictor's primary weights.",
    )
    parser.add_argument(
        "--out-models-dir",
        required=True,
        help="New predictor directory to create; it must not already exist.",
    )
    return parser


def _resolve_checkpoint(source, relative_path):
    relative = Path(str(relative_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "Checkpoint path must be relative and remain inside the predictor: "
            "%s" % relative_path)
    result = (source / relative).resolve()
    if source != result and source not in result.parents:
        raise ValueError(
            "Checkpoint path escapes predictor directory: %s" % relative_path)
    return result


def materialize(models_dir, policy, out_models_dir):
    """Create a predictor whose primary files use ``policy`` checkpoints."""
    source = Path(models_dir).resolve()
    out = Path(out_models_dir).resolve()
    manifest_path = source / "manifest.csv"
    if not manifest_path.is_file():
        raise ValueError("Missing predictor manifest: %s" % manifest_path)
    if out.exists():
        raise ValueError("Output already exists: %s" % out)
    if out == source or source in out.parents:
        raise ValueError("Output cannot be inside the source predictor directory")

    manifest = pandas.read_csv(manifest_path)
    checkpoint_column = "checkpoint_%s_weights" % policy
    if checkpoint_column not in manifest.columns:
        raise ValueError(
            "Predictor has no retained %s checkpoints (missing manifest "
            "column %s)" % (policy, checkpoint_column))

    checkpoints = []
    for _, row in manifest.iterrows():
        relative_path = row[checkpoint_column]
        if pandas.isna(relative_path):
            raise ValueError(
                "Model %s has no retained %s checkpoint" % (
                    row.model_name, policy))
        checkpoint = _resolve_checkpoint(source, relative_path)
        if not checkpoint.is_file():
            raise ValueError(
                "Missing retained %s checkpoint for %s: %s" % (
                    policy, row.model_name, checkpoint))
        checkpoints.append((row.model_name, checkpoint))

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.parent / (".%s.tmp-%s" % (out.name, uuid.uuid4().hex))
    try:
        shutil.copytree(source, temporary)
        provenance_models = []
        for model_name, checkpoint in checkpoints:
            destination = Path(Class1AffinityPredictor.weights_path(
                str(temporary), model_name))
            shutil.copy2(checkpoint, destination)
            digest = _sha256(checkpoint)
            provenance_models.append({
                "model_name": model_name,
                "checkpoint_sha256": digest,
                "primary_weights_sha256": _sha256(destination),
            })

        manifest["primary_checkpoint_policy"] = policy
        updated_configs = []
        for raw_config in manifest.config_json:
            config = json.loads(raw_config)
            config["materialized_checkpoint_policy"] = policy
            fit_info = config.get("fit_info") or []
            if fit_info:
                fit_info[-1]["materialized_checkpoint_policy"] = policy
            updated_configs.append(json.dumps(config))
        manifest["config_json"] = updated_configs
        manifest.to_csv(temporary / "manifest.csv", index=False)

        removed_stale_metadata = []
        for name in ("percent_ranks.csv", "optimization_info.json"):
            path = temporary / name
            if path.exists():
                path.unlink()
                removed_stale_metadata.append(name)

        provenance = {
            "format": 1,
            "source_models_dir": str(source),
            "source_manifest_sha256": _sha256(manifest_path),
            "checkpoint_policy": policy,
            "models": provenance_models,
            "removed_stale_metadata": removed_stale_metadata,
        }
        with (temporary / "checkpoint_selection.json").open("w") as fd:
            json.dump(provenance, fd, indent=2, sort_keys=True)
            fd.write("\n")
        temporary.replace(out)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return out


def run_argv(argv=None, prog="mhcflurry train materialize-affinity-checkpoint"):
    args = make_parser(prog).parse_args(argv)
    out = materialize(args.models_dir, args.policy, args.out_models_dir)
    print("Materialized %s affinity checkpoints to: %s" % (args.policy, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_argv())
