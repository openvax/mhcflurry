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

"""
Tests for measurement-driven per-worker GPU memory estimates.

The two ``assert`` blocks marked SANITY ANCHOR pin the estimators to values
measured on the release pan-allele config; if you change the formulas, update
these against fresh measurements rather than loosening them blindly.
"""

import json

import pandas

from mhcflurry.device_footprint import (
    estimate_affinity_calibration_device_worker_gb as calibration_gb,
    estimate_affinity_training_device_worker_gb as training_gb,
    estimate_device_worker_gb,
)

# Release pan-allele architecture: 10 networks, BLOSUM62 (21-wide), max_length
# 15, post-merge dense [1024, 512], no peptide dense layers, minibatch 128.
RELEASE_HYPERPARAMETERS = {
    "peptide_dense_layer_sizes": [],
    "layer_sizes": [1024, 512],
    "loss": "custom:mse_with_inequalities",
    "minibatch_size": 128,
    "peptide_encoding": {"max_length": 15, "vector_encoding_name": "BLOSUM62"},
}
NO_LEGACY_VALIDATION_HYPERPARAMETERS = dict(
    RELEASE_HYPERPARAMETERS,
    loss="mse",
)
RELEASE_NUM_NETWORKS = 10
RELEASE_TRAINING_ROWS = 249802         # curated affinity rows
RELEASE_CALIBRATION_ROWS = 800000      # 1e5/length x 8 lengths
RC14_TRAINING_ROWS = 994948            # raw rows before train command filters


def _write_manifest(tmp_path, n_networks, hyperparameters):
    config = json.dumps({"hyperparameters": hyperparameters})
    pandas.DataFrame({
        "model_name": ["model_%d" % i for i in range(n_networks)],
        "config_json": [config] * n_networks,
    }).to_csv(tmp_path / "manifest.csv", index=False)


# --------------------------------------------------------------------------
# Affinity calibration
# --------------------------------------------------------------------------

def test_calibration_sanity_anchor(tmp_path):
    # SANITY ANCHOR: 800k-row universe x 10-net ensemble. The selected
    # release ensemble uses the merged fast path, whose cached peptide-stage
    # tensor is wider than the raw BLOSUM62 peptide vector in manifest.csv.
    _write_manifest(tmp_path, RELEASE_NUM_NETWORKS, RELEASE_HYPERPARAMETERS)
    gb = calibration_gb(str(tmp_path), RELEASE_CALIBRATION_ROWS)
    assert 30.0 < gb < 33.0, gb


def test_calibration_scales_and_falls_back(tmp_path):
    _write_manifest(tmp_path, RELEASE_NUM_NETWORKS, RELEASE_HYPERPARAMETERS)
    small = calibration_gb(str(tmp_path), 8000)
    prod = calibration_gb(str(tmp_path), RELEASE_CALIBRATION_ROWS)
    huge = calibration_gb(str(tmp_path), 8000000)
    assert small < prod < huge
    assert small == 4.0                # tiny jobs floor at the minimum
    assert huge > 24.0                 # huge jobs dwarf the old static 24 GB

    # Explicit peptide-stage layer overrides the encoding-derived width.
    _write_manifest(tmp_path, RELEASE_NUM_NETWORKS, {
        "peptide_dense_layer_sizes": [64, 32],
        "peptide_encoding": {
            "max_length": 15, "vector_encoding_name": "BLOSUM62"},
    })
    assert calibration_gb(str(tmp_path), RELEASE_CALIBRATION_ROWS) < prod

    # Unreadable model / empty job -> None (planner uses the profile default).
    assert calibration_gb(str(tmp_path / "missing"), 800000) is None
    assert calibration_gb(str(tmp_path), 0) is None


def test_calibration_rc14_a100_40gb_worker_cap(tmp_path, monkeypatch):
    """Calibration should not pack multiple caches onto A100-40GB."""
    from mhcflurry.parallelism import auto_max_workers_per_gpu

    _write_manifest(tmp_path, RELEASE_NUM_NETWORKS, RELEASE_HYPERPARAMETERS)
    worker_gb = calibration_gb(str(tmp_path), 400000)
    assert 14.0 < worker_gb < 18.0, worker_gb

    monkeypatch.setenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "40")
    monkeypatch.setenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP", "4")
    assert auto_max_workers_per_gpu(
        num_jobs="auto",
        num_gpus=4,
        per_worker_gb=worker_gb,
    ) == 1


# --------------------------------------------------------------------------
# Affinity training
# --------------------------------------------------------------------------

def test_training_sanity_anchor():
    # SANITY ANCHOR: ~250k rows, minibatch 128. The estimate is above the
    # measured 1.85-2.4 GB steady-state minibatch peak because the release
    # pretrain path validates an inequality loss in one full-fold forward pass.
    gb = training_gb(RELEASE_HYPERPARAMETERS, RELEASE_TRAINING_ROWS)
    assert 9.0 < gb < 13.0, gb


def test_training_is_base_dominated_for_index_encoded_peptides():
    # Peptides are int8 indices device-resident (the default), so the dataset
    # barely affects the minibatch-only footprint at normal scale: it is
    # base-dominated unless the legacy inequality validation path is active.
    release = training_gb(
        NO_LEGACY_VALIDATION_HYPERPARAMETERS, RELEASE_TRAINING_ROWS)
    five_million = training_gb(
        NO_LEGACY_VALIDATION_HYPERPARAMETERS, 5000000)
    assert abs(five_million - release) < 1.0           # ~flat with dataset size

    # Only an extreme row count materially raises the resident int8 term.
    assert training_gb(
        NO_LEGACY_VALIDATION_HYPERPARAMETERS, 50000000) > release + 1.0

    # Batch size scales the (per-batch) activation term.
    assert (training_gb(
        NO_LEGACY_VALIDATION_HYPERPARAMETERS,
        RELEASE_TRAINING_ROWS,
        minibatch_size=32768) > release)

    # Missing hyperparameters / rows -> None.
    assert training_gb(None, RELEASE_TRAINING_ROWS) is None
    assert training_gb(RELEASE_HYPERPARAMETERS, 0) is None


def test_training_rc14_a100_40gb_worker_cap(monkeypatch):
    """The rc14 release recipe should not pack 4 workers onto A100-40GB."""
    from mhcflurry.parallelism import auto_max_workers_per_gpu

    rc14_small = dict(
        RELEASE_HYPERPARAMETERS,
        layer_sizes=[512, 256],
        minibatch_size=1024,
    )
    worker_gb = training_gb(rc14_small, RC14_TRAINING_ROWS)
    assert 23.0 < worker_gb < 27.0, worker_gb

    monkeypatch.setenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_FREE_VRAM_GB", "40")
    monkeypatch.setenv(
        "MHCFLURRY_AUTO_MAX_WORKERS_PER_GPU_HARD_CAP", "4")
    assert auto_max_workers_per_gpu(
        num_jobs="auto",
        num_gpus=4,
        per_worker_gb=worker_gb,
    ) == 1


# --------------------------------------------------------------------------
# Unified dispatcher
# --------------------------------------------------------------------------

def test_dispatcher_routes_by_workload(tmp_path):
    from mhcflurry.workload_planning import (
        WORKLOAD_AFFINITY_CALIBRATION,
        WORKLOAD_AFFINITY_TRAINING,
        WORKLOAD_AFFINITY_SELECTION,
    )

    _write_manifest(tmp_path, RELEASE_NUM_NETWORKS, RELEASE_HYPERPARAMETERS)
    assert estimate_device_worker_gb(
        WORKLOAD_AFFINITY_CALIBRATION,
        {"prediction_rows": RELEASE_CALIBRATION_ROWS},
        models_dir=str(tmp_path)) == calibration_gb(
            str(tmp_path), RELEASE_CALIBRATION_ROWS)

    assert estimate_device_worker_gb(
        WORKLOAD_AFFINITY_TRAINING,
        {"training_rows": RELEASE_TRAINING_ROWS,
         "hyperparameters": RELEASE_HYPERPARAMETERS}) == training_gb(
            RELEASE_HYPERPARAMETERS, RELEASE_TRAINING_ROWS)

    # Workloads without an estimator fall through to the static profile default.
    assert estimate_device_worker_gb(
        WORKLOAD_AFFINITY_SELECTION, {}, models_dir=str(tmp_path)) is None
