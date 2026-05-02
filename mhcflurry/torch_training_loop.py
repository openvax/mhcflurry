"""Shared PyTorch training-loop helpers used by multiple mhcflurry trainers.

These helpers are model-agnostic: they assume only that the caller has a
``torch.nn.Module`` network and a ``torch.device``. The pure-helper subset
extracted here is the foundation for unifying perf-critical paths
(async H2D, prefetcher, batched validation) between
``Class1NeuralNetwork`` (affinity) and ``Class1ProcessingNeuralNetwork``
in follow-up changes.

Anything that touches affinity-specific machinery (random-negative
resampling, multi-output / inequality losses, allele encoding,
percentile-rank calibration) stays in ``class1_neural_network.py``.
"""

import logging
import os

import torch

_TRITON_AUTOGRAD_WARMED_DEVICES = set()


def _configure_matmul_precision(device):
    """Optionally enable TF32 + cuDNN benchmark on CUDA Ampere+.

    Both are runtime settings with no JIT/startup overhead, but TF32 changes
    CUDA matmul numerics. Leave PyTorch's default behavior untouched unless
    the caller explicitly opts in with ``MHCFLURRY_MATMUL_PRECISION``.

    **TF32 (``torch.set_float32_matmul_precision``)** — changes matmul
    kernel selection. On Ampere+ (A100/H100/L40S/...) it can be ~2×
    faster for matmul-heavy paths, with fp32 accumulation preserved but
    input-mantissa truncation.

    **cuDNN benchmark** — tells cuDNN to search for the fastest
    algorithm for each (shape, dtype, stride) tuple on first call and
    cache the result. Useful when input shapes are stable across
    iterations (our fit-generator + fit paths guarantee this via
    drop_last + fixed-size pretrain chunks). One-time cost of
    benchmarking on the first forward pass (~1-2 s), then every
    subsequent call hits the cached best algorithm. mhcflurry's MLP
    doesn't use Conv layers in the default architecture, so the gain
    is modest here — but it's free and lets convolutional variants
    (if ever configured via ``locally_connected_layers``) benefit.

    Backend interactions:
    - CUDA Ampere+: TF32 enabled + cudnn.benchmark on.
    - CUDA pre-Ampere (V100/T4): TF32 is no-op; cudnn.benchmark still helps.
    - CPU: both are no-ops.
    - MPS: both are no-ops.

    Opt in via ``MHCFLURRY_MATMUL_PRECISION={highest,high,medium}``.
    ``highest`` keeps full fp32 precision while still enabling
    ``cudnn.benchmark``.
    """
    if device.type != "cuda":
        return
    precision = os.environ.get("MHCFLURRY_MATMUL_PRECISION")
    if not precision:
        return
    torch.set_float32_matmul_precision(precision)
    # cuDNN benchmark is cheap to enable and has no effect if the
    # workload never triggers a cuDNN kernel (plain Linear + RMSprop
    # MLP). Guarded against environments that disabled cuDNN entirely.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def _maybe_compile_network(network, device):
    """Wrap ``network`` with ``torch.compile`` when the env asks for it.

    Gated on ``MHCFLURRY_TORCH_COMPILE=1`` and a CUDA device.
    ``torch.compile`` is heavy: JIT graph capture + kernel fusion + on-
    disk cache + recompile-on-shape-change. The TF32 knob is cheaper
    and independent — see ``_configure_matmul_precision``.

    ``MHCFLURRY_TORCH_COMPILE_MODE`` picks the ``mode=`` kwarg (default,
    reduce-overhead, max-autotune). Default is "default" — codegen time
    is already heavy without max-autotune, and our shape-stable batching
    is what unlocks the big wins regardless of mode.

    Returns the compiled module (an ``OptimizedModule`` that forwards
    ``.train()``, ``.eval()``, ``.state_dict()``, ``.parameters()`` to
    the wrapped original) so call sites can swap it in without
    threading a second reference through the training loop.

    Compilation cost: first forward pass triggers graph capture +
    codegen (typically 30 s – 2 min on a 2-layer MLP like ours).
    Subsequent calls with the same input shapes hit the in-process
    cache; subsequent *processes* hit the on-disk cache
    (``~/.cache/torch``) as long as the graph matches.
    """
    if device.type != "cuda":
        return network
    if os.environ.get("MHCFLURRY_TORCH_COMPILE", "0") != "1":
        return network
    # Idempotent: if ``network`` is already an OptimizedModule (i.e.
    # we've been called before on the same instance, from inside an
    # epoch loop), return it unchanged.
    if hasattr(network, "_orig_mod"):
        return network
    mode = os.environ.get("MHCFLURRY_TORCH_COMPILE_MODE", "default")
    # ``dynamic=True`` tells dynamo to generate one shape-polymorphic
    # graph instead of specializing on every batch shape it sees.
    # mhcflurry's forward is called with at least three distinct row
    # counts per work item — pretrain (64 rows), finetune (128), and
    # validation (4× finetune = 512) — so dynamic=False triggers a
    # recompile storm (8+ specializations observed in stderr with
    # [0/8] from torch._dynamo.convert_frame). Each recompile is a
    # 10-30 s codegen pass, defeating the point. Dynamic mode costs a
    # few % on the individual kernel but avoids paying the storm.
    # Override with MHCFLURRY_TORCH_COMPILE_DYNAMIC=0 for static mode
    # if a caller can guarantee single-shape input.
    dynamic = os.environ.get("MHCFLURRY_TORCH_COMPILE_DYNAMIC", "1") != "0"
    logging.info("torch.compile enabled (mode=%s, dynamic=%s)", mode, dynamic)
    return torch.compile(network, mode=mode, dynamic=dynamic)


def _maybe_compile_loss(loss_obj, device):
    """Wrap a loss module with ``torch.compile`` when the env asks for it.

    Gated on ``MHCFLURRY_TORCH_COMPILE=1`` and a CUDA device — same
    criteria as ``_maybe_compile_network``. ``MSEWithInequalities``
    issues ~10 small elementwise kernels per step in eager mode
    (reshape → subtract → compare → cast → multiply → compare → cast →
    multiply → square → sum), each with its own launch overhead that
    adds up to meaningful wall-clock on A100 with a sub-ms compute
    budget. Dynamo fuses those into a couple of kernels and cuts the
    loss's share of step time to near-zero.

    Dispatch via ``MHCFLURRY_TORCH_COMPILE_LOSS_MODE`` (falls back to
    ``MHCFLURRY_TORCH_COMPILE_MODE``) so the loss's compile mode can
    be tuned independently of the network's — loss ops are tiny and
    "reduce-overhead" makes less sense than on the full forward pass.

    Idempotent: a second call on an already-wrapped loss returns it
    unchanged.
    """
    if device.type != "cuda":
        return loss_obj
    if os.environ.get("MHCFLURRY_TORCH_COMPILE", "0") != "1":
        return loss_obj
    # Loss compilation is enabled by default when network compilation is
    # enabled. PyTorch 2.4 / Triton 3.0 has an upstream bug where the first
    # Triton kernel launched from autograd's backward worker thread can fail
    # with ``RuntimeError: Triton Error [CUDA]: invalid device context``.
    # Running a tiny CUDA backward in the *training worker process* initializes
    # that thread-local CUDA context before the compiled loss backward fires.
    if os.environ.get("MHCFLURRY_TORCH_COMPILE_LOSS", "1") != "1":
        return loss_obj
    if hasattr(loss_obj, "_orig_mod"):
        return loss_obj
    mode = os.environ.get(
        "MHCFLURRY_TORCH_COMPILE_LOSS_MODE",
        os.environ.get("MHCFLURRY_TORCH_COMPILE_MODE", "default"),
    )
    # Loss takes (y_pred, y_true) and optionally sample_weights with
    # dynamic-batch shapes that mirror the network's forward. Match the
    # network's dynamic/static policy via the same env knob.
    dynamic = os.environ.get("MHCFLURRY_TORCH_COMPILE_DYNAMIC", "1") != "0"
    _warm_cuda_autograd_for_triton(device)
    logging.info("torch.compile applied to loss (mode=%s, dynamic=%s)", mode, dynamic)
    return torch.compile(loss_obj, mode=mode, dynamic=dynamic)


def _warm_cuda_autograd_for_triton(device):
    """Initialize CUDA context in autograd's backward thread once per device."""
    if device.type != "cuda":
        return
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    key = int(index)
    if key in _TRITON_AUTOGRAD_WARMED_DEVICES:
        return
    with torch.cuda.device(device):
        torch.empty(1, device=device, requires_grad=True).sum().backward()
        torch.cuda.synchronize(device)
    _TRITON_AUTOGRAD_WARMED_DEVICES.add(key)


def _effective_validation_batch_size(
        device, configured_batch_size, minibatch_size,
        model=None, num_workers_per_gpu=1):
    """Return the validation batch size to use for the current device.

    Static heuristic. MUST be deterministic across calls — fit() /
    fit_streaming_batches() call this per-epoch, and torch.compile caches
    specializations by input shape. A validation batch that varies
    with live free-VRAM (the auto-sized approach) forces the
    compiled graph to re-codegen every epoch and with 16 training
    workers × 32 inductor compile workers on a 128-vCPU box pins the
    CPU at hundreds of concurrent compile jobs — observed to stall
    training indefinitely. The auto-sized prediction batch size in
    ``compute_prediction_batch_size`` is fine for mhcflurry-predict
    where each call is a single forward; training-time validation is
    not that shape.

    ``model`` and ``num_workers_per_gpu`` kwargs are retained for API
    compatibility with the call sites but are intentionally unused.
    """
    del model, num_workers_per_gpu
    if configured_batch_size:
        return int(configured_batch_size)
    if device.type == "cuda":
        # Validation is forward-only and the networks are tiny relative
        # to modern GPU memory. A much larger default batch dramatically
        # cuts kernel-launch / Python-loop overhead versus 4 *
        # minibatch_size, while staying deterministic across epochs.
        return max(4 * minibatch_size, 4096)
    return 4 * minibatch_size
