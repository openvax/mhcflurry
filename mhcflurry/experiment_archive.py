# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create portable, plot-reconstructable snapshots of training experiments."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

import pandas


SCHEMA_VERSION = 1
DEFAULT_MAX_COPY_BYTES = 64 * 1024 * 1024
HASH_LINE_RE = re.compile(r"^([0-9a-f]{64})\s+(.+)$")


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA256 digest of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda: fd.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if pandas.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path, rows, preferred_fields=()):
    path = Path(path)
    all_fields = {key for row in rows for key in row}
    fieldnames = [field for field in preferred_fields if field in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))
    with path.open("w", newline="") as fd:
        if not fieldnames:
            return
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _model_manifest_paths(source_dir):
    result = []
    for path in sorted(Path(source_dir).rglob("manifest.csv")):
        with path.open(newline="") as fd:
            fieldnames = next(csv.reader(fd), [])
        if "model_name" in fieldnames and "config_json" in fieldnames:
            result.append(path)
    return result


def export_training_tables(source_dir, out_dir):
    """Export normalized model, fit, and per-epoch tables from manifests.

    The source manifests remain authoritative. These tables deliberately omit
    serialized network graphs and weights so ordinary plotting and audit code
    can consume the histories without loading model implementations.
    """
    source_dir = Path(source_dir).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_rows = []
    fit_rows = []
    epoch_rows = []
    config_records = []

    for manifest_path in _model_manifest_paths(source_dir):
        relative_manifest = manifest_path.relative_to(source_dir).as_posix()
        manifest = pandas.read_csv(manifest_path)
        for _, manifest_row in manifest.iterrows():
            config = json.loads(manifest_row["config_json"])
            hyperparameters = config.get("hyperparameters", {}) or {}
            fit_infos = config.get("fit_info", []) or []
            model_name = str(manifest_row["model_name"])
            config_payload = json.dumps(
                {
                    "hyperparameters": hyperparameters,
                    "fit_info": fit_infos,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            model_rows.append({
                "manifest_path": relative_manifest,
                "model_name": model_name,
                "allele": manifest_row.get("allele"),
                "config_sha256": hashlib.sha256(
                    str(manifest_row["config_json"]).encode()).hexdigest(),
                "fit_count": len(fit_infos),
                "topology": hyperparameters.get("topology"),
                "layer_sizes": json.dumps(
                    hyperparameters.get("layer_sizes", []),
                    separators=(",", ":"),
                ),
                "activation": hyperparameters.get("activation"),
                "convolutional_activation": hyperparameters.get(
                    "convolutional_activation"),
                "normalization": hyperparameters.get("normalization", "none"),
                "dropout_keep_probability": hyperparameters.get(
                    "dropout_probability"),
                "dropout_rate": hyperparameters.get("dropout_rate"),
                "restore_best_weights": hyperparameters.get(
                    "restore_best_weights", False),
                "patience": hyperparameters.get("patience"),
                "learning_rate": hyperparameters.get("learning_rate"),
                "minibatch_size": hyperparameters.get("minibatch_size"),
                "optimizer": hyperparameters.get("optimizer"),
                "optimizer_implementation": hyperparameters.get(
                    "optimizer_implementation"),
                "initializer": hyperparameters.get("init"),
                "data_dependent_initialization_method": hyperparameters.get(
                    "data_dependent_initialization_method"),
                "data_dependent_initialization_target": hyperparameters.get(
                    "data_dependent_initialization_target"),
                "convolutional_filters": hyperparameters.get(
                    "convolutional_filters"),
                "convolutional_kernel_size": hyperparameters.get(
                    "convolutional_kernel_size"),
                "n_flank_length": hyperparameters.get("n_flank_length"),
                "c_flank_length": hyperparameters.get("c_flank_length"),
                "hyperparameters_json": json.dumps(
                    hyperparameters, sort_keys=True, separators=(",", ":")),
            })
            config_records.append({
                "manifest_path": relative_manifest,
                "model_name": model_name,
                "config": json.loads(config_payload),
            })

            for fit_index, fit_info in enumerate(fit_infos):
                if not isinstance(fit_info, dict):
                    continue
                training_info = fit_info.get("training_info", {}) or {}
                identity = {
                    "manifest_path": relative_manifest,
                    "model_name": model_name,
                    "fit_index": fit_index,
                    "phase": training_info.get("phase", "unknown"),
                    "fold": training_info.get("fold_num"),
                    "architecture": training_info.get("architecture_num"),
                    "replicate": training_info.get("replicate_num"),
                    "work_item_name": training_info.get("work_item_name"),
                }
                scalar_info = {
                    "fit_" + key: _json_value(value)
                    for key, value in fit_info.items()
                    if key != "training_info" and not isinstance(value, list)
                }
                scalar_training_info = {
                    "training_" + key: _json_value(value)
                    for key, value in training_info.items()
                    if not isinstance(value, list)
                }
                fit_rows.append({
                    **identity,
                    **scalar_info,
                    **scalar_training_info,
                    "fit_info_json": json.dumps(
                        fit_info, sort_keys=True, separators=(",", ":")),
                })

                series = {
                    key: value
                    for key, value in fit_info.items()
                    if isinstance(value, list)
                }
                epoch_count = max((len(value) for value in series.values()), default=0)
                for epoch_index in range(epoch_count):
                    row = {**identity, "epoch": epoch_index + 1}
                    for key, values in series.items():
                        if epoch_index < len(values):
                            row[key] = _json_value(values[epoch_index])
                    epoch_rows.append(row)

    preferred = (
        "manifest_path", "model_name", "fit_index", "phase", "fold",
        "architecture", "replicate", "work_item_name", "epoch",
    )
    _write_csv(out_dir / "models.csv", model_rows, preferred[:2])
    _write_csv(out_dir / "fits.csv", fit_rows, preferred[:-1])
    _write_csv(out_dir / "training_history.csv", epoch_rows, preferred)
    with (out_dir / "model_configs.jsonl").open("w") as fd:
        for record in config_records:
            fd.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "model_manifest_count": len(_model_manifest_paths(source_dir)),
        "model_count": len(model_rows),
        "fit_count": len(fit_rows),
        "epoch_count": len(epoch_rows),
    }


def _artifact_role(relative_path):
    path = Path(relative_path)
    name = path.name
    if name.startswith("weights_") and path.suffix == ".npz":
        return "weight"
    if name == "train_data.csv.bz2":
        return "training_data"
    if name == "manifest.csv" and "models" in relative_path:
        return "model_manifest"
    if name == "gpu_occupancy.csv":
        return "telemetry"
    if name.endswith(("predictions.csv", "predictions.csv.bz2")):
        return "prediction"
    if name.startswith("benchmark.") and name.endswith((".csv", ".csv.bz2")):
        return "prediction"
    if path.suffix.lower() in (".pdf", ".png", ".svg"):
        return "figure"
    if name.endswith((".log", ".txt")) and (
            "log" in name.lower() or name.startswith("LOG-worker")):
        return "log"
    if "condition" in path.parts and path.suffix in (".yaml", ".yml"):
        return "configuration"
    if name.startswith("provenance") or name in {
            "manifest.json", "manifest.csv", "command.sh",
            "started_at_utc.txt", "completed_at_utc.txt",
            "queue_provenance.txt", "verification.json"}:
        return "provenance"
    if name in {
            "summary.csv", "summary.json", "summary.md", "summary_table.csv",
            "release_summary.csv", "release_summary.md", "per_allele.csv",
            "per_length.csv", "per_length_per_allele.csv",
            "training_overlap.json", "model_selection_summary.csv.bz2"}:
        return "metric"
    if name == "predictor_info.csv" or name.startswith("accuracy_scores."):
        return "metric"
    return "other"


def _copy_role(role):
    return role in {
        "weight", "training_data", "model_manifest", "telemetry", "log",
        "configuration", "provenance", "metric", "prediction", "figure",
    }


def _copy_regardless_of_size(role):
    return role == "prediction"


def _copy_artifact(source, target, prefer_hardlink=False):
    """Copy an artifact, or hard-link immutable large data when possible."""
    if prefer_hardlink:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copy"


def _environment_record():
    distributions = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            distributions[name] = distribution.version
    gpu_query = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            gpu_query = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted(distributions.items(), key=lambda item: item[0].lower())),
        "gpus": gpu_query,
    }


def _external_hash_records(source_dir):
    records = []
    for provenance in sorted(Path(source_dir).rglob("*provenance*")):
        if not provenance.is_file() or provenance.stat().st_size > 8 * 1024 * 1024:
            continue
        try:
            lines = provenance.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            match = HASH_LINE_RE.match(line.strip())
            if match:
                records.append({
                    "recorded_sha256": match.group(1),
                    "path": match.group(2),
                    "recorded_in": provenance.relative_to(source_dir).as_posix(),
                })
    return records


def _slug(value):
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not result:
        raise ValueError("Experiment name must contain a letter or number")
    return result


def _read_source_commit(source_dir, explicit):
    if explicit:
        return explicit.strip()
    provenance = Path(source_dir) / "provenance.txt"
    if provenance.exists():
        for line in provenance.read_text(errors="replace").splitlines():
            if line.startswith("source_commit="):
                return line.split("=", 1)[1].strip()
    raise ValueError(
        "source_commit is required when source_dir/provenance.txt does not "
        "record source_commit=<hash>")


def _collector_commit():
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def snapshot_experiment(
    source_dir,
    experiments_dir,
    name,
    *,
    source_commit=None,
    collector_commit=None,
    source_archive=None,
    command_files=(),
    input_files=(),
    captured_at=None,
    max_copy_bytes=DEFAULT_MAX_COPY_BYTES,
):
    """Create and return one immutable timestamped experiment snapshot."""
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise ValueError("source_dir is not a directory: %s" % source_dir)
    source_commit = _read_source_commit(source_dir, source_commit)
    if not source_commit:
        raise ValueError("source_commit must not be empty")
    captured_at = captured_at or datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    captured_at = captured_at.astimezone(timezone.utc)
    timestamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    snapshot_name = "%s-%s-%s" % (
        timestamp, _slug(name), _slug(source_commit)[:12])
    experiments_dir = Path(experiments_dir).resolve()
    experiments_dir.mkdir(parents=True, exist_ok=True)
    destination = experiments_dir / snapshot_name
    temporary = experiments_dir / ("." + snapshot_name + ".tmp")
    if destination.exists() or temporary.exists():
        raise FileExistsError("Experiment snapshot already exists: %s" % destination)
    temporary.mkdir()
    artifacts_dir = temporary / "artifacts"
    artifacts_dir.mkdir()
    data_dir = temporary / "data"
    data_dir.mkdir()

    inventory = []
    for source_path in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative_path = source_path.relative_to(source_dir)
        role = _artifact_role(relative_path.as_posix())
        size = source_path.stat().st_size
        copied = _copy_role(role) and (
            size <= max_copy_bytes or _copy_regardless_of_size(role))
        storage = "inventory_only"
        if copied:
            target = artifacts_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            storage = _copy_artifact(
                source_path,
                target,
                prefer_hardlink=role == "prediction",
            )
        inventory.append({
            "relative_path": relative_path.as_posix(),
            "role": role,
            "size_bytes": size,
            "sha256": sha256_file(source_path),
            "copied": copied,
            "storage": storage,
        })

    training_tables = export_training_tables(source_dir, data_dir)
    _write_csv(
        temporary / "source_files.csv",
        inventory,
        (
            "relative_path", "role", "size_bytes", "sha256", "copied",
            "storage",
        ),
    )
    _write_csv(
        temporary / "external_inputs.csv",
        _external_hash_records(source_dir),
        ("path", "recorded_sha256", "recorded_in"),
    )

    supplemental = []
    for role, paths in (("command", command_files), ("input", input_files)):
        resolved_paths = [Path(value).resolve() for value in paths]
        basenames = [path.name for path in resolved_paths]
        disambiguate = len(basenames) != len(set(basenames))
        for index, path in enumerate(resolved_paths):
            if not path.is_file():
                raise ValueError("%s file does not exist: %s" % (role, path))
            record = {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "copied": path.stat().st_size <= max_copy_bytes,
            }
            if record["copied"]:
                target_name = (
                    "%03d-%s" % (index, path.name)
                    if disambiguate else path.name
                )
                target = temporary / "supplemental" / role / target_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                record["snapshot_path"] = target.relative_to(temporary).as_posix()
            supplemental.append(record)

    source_archive_record = None
    if source_archive:
        archive = Path(source_archive).resolve()
        if not archive.is_file():
            raise ValueError("source_archive is not a file: %s" % archive)
        target = temporary / "source" / archive.name
        target.parent.mkdir()
        shutil.copy2(archive, target)
        source_archive_record = {
            "path": str(archive),
            "snapshot_path": target.relative_to(temporary).as_posix(),
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }

    experiment = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "captured_at_utc": captured_at.isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "collector_commit": collector_commit or _collector_commit(),
        "source_dir": str(source_dir),
        "source_archive": source_archive_record,
        "training_tables": training_tables,
        "source_file_count": len(inventory),
        "copied_source_file_count": sum(row["copied"] for row in inventory),
        "supplemental_files": supplemental,
    }
    (temporary / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n")
    (temporary / "environment.json").write_text(
        json.dumps(_environment_record(), indent=2, sort_keys=True) + "\n")
    (temporary / "README.md").write_text(
        "\n".join([
            "# %s" % name,
            "",
            "- Captured: `%s`" % experiment["captured_at_utc"],
            "- Training source commit: `%s`" % source_commit,
            "- Snapshot collector commit: `%s`" % (
                experiment["collector_commit"] or "unavailable"),
            "- Original output: `%s`" % source_dir,
            "",
            "`data/training_history.csv` is the tidy per-epoch table used to "
            "rebuild learning-curve plots. `data/models.csv`, `data/fits.csv`, "
            "and `data/model_configs.jsonl` preserve model and fit metadata. "
            "Comparison tables, telemetry, configs, manifests, and logs are "
            "under `artifacts/`. Held-out prediction tables and generated "
            "figures are copied there even when prediction files exceed the "
            "ordinary artifact-size threshold. Same-filesystem prediction "
            "tables use hard links to avoid duplicate storage; their storage "
            "mode is recorded in `source_files.csv`.",
            "",
            "`source_files.csv` inventories every original artifact by SHA256. "
            "Weights and per-model training tables below the configured copy-size "
            "limit are retained; larger files remain checksummed in the inventory. "
            "`external_inputs.csv` preserves hashes recorded by the runner. "
            "`environment.json` records the capture environment.",
            "",
            "To reconstruct, check out the training source commit, verify input "
            "and weight hashes against the inventories, then use a preserved "
            "command file from `supplemental/command/` or `artifacts/command.sh`.",
            "",
            "Rebuild learning curves with `mhcflurry train plot-loss-curves` "
            "against a copied model manifest. Rebuild comparison plots and a "
            "review PDF with `mhcflurry plot-model-comparison --input "
            "artifacts/<comparison> --summary-pdf <output.pdf>`.",
            "",
        ]))
    temporary.rename(destination)
    return destination
