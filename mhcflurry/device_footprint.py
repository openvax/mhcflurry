"""
Measurement-driven per-worker GPU memory estimates.

The orchestrator decides how many workers to pack on each GPU from a per-worker
VRAM estimate (``device_worker_gb``). Historically that was a static constant in
the workload profile (e.g. 24 GB for calibration, 4 GB for training) — a value
hand-tuned for one production config that is wrong for jobs / networks of other
sizes. This module estimates it instead, from the model architecture (read from
``manifest.csv`` — no weights) and the job size, so the worker count adapts.

Every estimator returns ``None`` whenever it can't read the model or lacks the
job size, so the planner falls back to the static profile default. Estimates are
deliberately conservative (a safety multiplier + a floor) so we never pack more
aggressively than the validated baseline.

Sanity anchors (release pan-allele config, live diagnostics from the 2026-04-28
``release_exact`` run):

  * Affinity TRAINING — ~250k curated rows, minibatch 128 — measured steady-state
    ~1.85-2.4 GB. Dominated by the device-resident peptide-vector tensor
    ``(N, L, V) float32`` (alleles are a shared index-embedding, so cheap) plus a
    fixed base (CUDA context + torch.compile workspace + weights + optimizer +
    embedding). ``estimate_affinity_training_device_worker_gb`` reproduces this.
  * Affinity CALIBRATION — 800k-row peptide universe, 10-net ensemble — the
    cached peptide-stage tensor is ~12 GB (the static profile assumed 24 GB).
    ``estimate_affinity_calibration_device_worker_gb`` reproduces this.

The per-row resident formulas track the genuinely-varying terms: full dataset /
peptide-universe row count, the peptide encoding dims (``max_length`` x feature
width, which differ by ``vector_encoding_name`` — BLOSUM62=21, the new
physchem/atchley/composite encodings differ), network width, ensemble size, and
batch size.
"""

import json
from os.path import join, exists

import pandas

_GIB = 1024.0 ** 3
_FLOAT32_BYTES = 4

# ---- Affinity calibration (cached peptide-stage tensor) --------------------
_CALIBRATION_DEFAULT_STAGE_DIM = 1024     # when the encoding can't be derived
_CALIBRATION_OVERHEAD_GB = 3.0            # model weights + activation headroom
_CALIBRATION_MIN_DEVICE_WORKER_GB = 4.0   # don't over-pack from a tiny estimate

# ---- Affinity training (device-resident dataset + activations) -------------
# Fixed per-worker cost that does NOT scale with the dataset: CUDA context,
# torch.compile inductor workspace, model weights + Adam optimizer state, and
# the shared allele index-embedding. Calibrated so the release config lands in
# the measured ~2 GB band; the residual after the resident + activation terms.
_TRAINING_BASE_GB = 1.5
# The device fit materializes real rows AND a random-negative slice in one
# combined buffer (~2x rows). Conservative.
_TRAINING_RANDOM_NEGATIVE_FACTOR = 2.0
# Forward activations + gradients + backward working set, as a multiple of the
# single-forward activation bytes.
_TRAINING_ACTIVATION_BACKWARD_FACTOR = 3.0
_TRAINING_DEFAULT_MINIBATCH = 128
_TRAINING_DEFAULT_MERGE_WIDTH = 1024      # when layer widths can't be derived
# Applied to the whole estimate; keeps margin over the measured peak.
_TRAINING_SAFETY_MARGIN = 1.3
# Never below the measured steady-state; prevents over-packing vs the baseline.
_TRAINING_MIN_DEVICE_WORKER_GB = 2.5


def _peptide_vector_encoding_width(name):
    """Per-amino-acid feature width of a vector encoding, or None if unknown."""
    if not name:
        return None
    try:
        from .amino_acid import ENCODING_DATA_FRAMES
        return int(ENCODING_DATA_FRAMES[name].shape[1])
    except Exception:
        return None


def _read_manifest(models_dir):
    """Return ``(num_networks, first_hyperparameters)`` from ``manifest.csv``.

    Reads only the manifest (no weights). Returns ``None`` if it can't be read.
    """
    manifest_path = join(models_dir, "manifest.csv")
    if not exists(manifest_path):
        return None
    try:
        manifest = pandas.read_csv(manifest_path)
    except Exception:
        return None
    if len(manifest) == 0:
        return None
    try:
        hyperparameters = json.loads(
            manifest.iloc[0]["config_json"]).get("hyperparameters", {})
    except Exception:
        hyperparameters = {}
    return max(len(manifest), 1), hyperparameters


def _peptide_stage_dim(hyperparameters):
    """Peptide-side activation width: the last peptide dense layer if set, else
    the flattened peptide vector encoding (max_length x feature width)."""
    dense = hyperparameters.get("peptide_dense_layer_sizes") or []
    if dense:
        return int(dense[-1])
    encoding = hyperparameters.get("peptide_encoding", {})
    max_length = int(encoding.get("max_length", 0) or 0)
    width = _peptide_vector_encoding_width(encoding.get("vector_encoding_name"))
    if max_length and width:
        return max_length * width
    return None


def _peptide_row_bytes(hyperparameters):
    """Bytes for one peptide's resident ``(L, V) float32`` encoding, or None."""
    encoding = hyperparameters.get("peptide_encoding", {})
    max_length = int(encoding.get("max_length", 0) or 0)
    width = _peptide_vector_encoding_width(encoding.get("vector_encoding_name"))
    if max_length and width:
        return max_length * width * _FLOAT32_BYTES
    return None


def _merge_width(hyperparameters):
    """Rough peak hidden-layer width for the activation estimate."""
    candidates = []
    for key in ("layer_sizes", "peptide_dense_layer_sizes",
                "allele_dense_layer_sizes"):
        candidates.extend(int(x) for x in (hyperparameters.get(key) or []))
    stage = _peptide_stage_dim(hyperparameters)
    if stage:
        candidates.append(int(stage))
    return max(candidates) if candidates else _TRAINING_DEFAULT_MERGE_WIDTH


def estimate_affinity_calibration_device_worker_gb(models_dir, prediction_rows):
    """Per-worker VRAM (GB) for affinity percentile-rank calibration, or None.

    Dominant consumer: the cached peptide-stage tensor (peptide-side activations
    precomputed for the whole peptide universe and reused across allele
    batches), ``~ num_networks x stage_dim x num_peptides x 4``.
    """
    if not prediction_rows:
        return None
    manifest = _read_manifest(models_dir)
    if manifest is None:
        return None
    num_networks, hyperparameters = manifest
    stage_dim = _peptide_stage_dim(hyperparameters) or _CALIBRATION_DEFAULT_STAGE_DIM
    cache_gb = (
        num_networks * int(stage_dim) * int(prediction_rows)
        * _FLOAT32_BYTES / _GIB
    )
    return max(
        _CALIBRATION_MIN_DEVICE_WORKER_GB,
        _CALIBRATION_OVERHEAD_GB + cache_gb)


def estimate_affinity_training_device_worker_gb(
        hyperparameters, num_rows, minibatch_size=None):
    """Per-worker VRAM (GB) for device-resident affinity training, or None.

    Takes the ``hyperparameters`` dict directly (the model doesn't exist yet at
    training time — it's read from the hyperparameters YAML the command loaded).
    Models the three terms that actually scale:

      * resident dataset tensor: the full fit's peptide vectors live on device as
        ``(N, L, V) float32``, plus a random-negative slice (~2x rows). Alleles
        are a shared index-embedding, so they are part of the fixed base.
      * batch activations: ``minibatch x merge_width x 4`` x a backward factor.
      * a fixed base (CUDA context + compile workspace + weights + optimizer +
        embedding) that does not scale with the dataset.

    A safety multiplier and a floor keep it conservative.
    """
    if not num_rows or hyperparameters is None:
        return None

    peptide_row_bytes = _peptide_row_bytes(hyperparameters)
    if peptide_row_bytes is None:
        # Index-encoded peptides (or an unknown encoding) are cheap to keep
        # resident; the base dominates. Fall back to the floor rather than a
        # bogus huge/small estimate.
        return _TRAINING_MIN_DEVICE_WORKER_GB

    minibatch = int(minibatch_size or hyperparameters.get("minibatch_size")
                    or _TRAINING_DEFAULT_MINIBATCH)
    merge_width = _merge_width(hyperparameters)

    resident_gb = (
        _TRAINING_RANDOM_NEGATIVE_FACTOR * int(num_rows) * peptide_row_bytes
        / _GIB
    )
    activation_gb = (
        minibatch * merge_width * _FLOAT32_BYTES
        * _TRAINING_ACTIVATION_BACKWARD_FACTOR / _GIB
    )
    footprint = (
        (_TRAINING_BASE_GB + resident_gb + activation_gb)
        * _TRAINING_SAFETY_MARGIN
    )
    return max(_TRAINING_MIN_DEVICE_WORKER_GB, footprint)


def estimate_device_worker_gb(workload_name, hints, models_dir=None):
    """Unified per-worker VRAM (GB) estimate, or None to use the profile default.

    Dispatches per workload. ``hints`` is a dict that may carry:
      * calibration: ``prediction_rows`` (+ ``models_dir`` for the trained model).
      * training: ``training_rows``, ``hyperparameters`` (the model isn't trained
        yet, so the architecture comes from the hyperparameters dict),
        ``minibatch_size``.
    """
    hints = hints or {}
    # Imported lazily to avoid a circular import at module load.
    from .workload_planning import (
        WORKLOAD_AFFINITY_CALIBRATION,
        WORKLOAD_AFFINITY_TRAINING,
    )
    if workload_name == WORKLOAD_AFFINITY_CALIBRATION:
        if models_dir is None:
            return None
        return estimate_affinity_calibration_device_worker_gb(
            models_dir, hints.get("prediction_rows"))
    if workload_name == WORKLOAD_AFFINITY_TRAINING:
        return estimate_affinity_training_device_worker_gb(
            hints.get("hyperparameters"), hints.get("training_rows"),
            minibatch_size=hints.get("minibatch_size"))
    return None
