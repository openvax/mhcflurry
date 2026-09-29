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

"""Shared PyTorch training helpers used by multiple mhcflurry trainers.

These helpers are model-agnostic: they assume only that the caller has a
``torch.nn.Module`` network and a ``torch.device``. They provide checkpoint copying, numerical settings, compilation and
validation sizing for ``Class1NeuralNetwork`` (affinity) and
``Class1ProcessingNeuralNetwork``.

Anything that touches affinity-specific machinery (random-negative
resampling, multi-output / inequality losses, allele encoding,
percentile-rank calibration) stays in the class I modules.
"""

import logging
import os

import torch

from .parallelism import resolve_torchinductor_compile_threads_env

_TRITON_AUTOGRAD_WARMED_DEVICES = set()


def copy_module_state_dict_to_cpu(module):
    """Return an independent CPU checkpoint of a module's complete state."""
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def configure_matmul_precision(device):
    """Apply an explicitly requested CUDA matmul precision and cuDNN autotuning.

    ``MHCFLURRY_MATMUL_PRECISION`` accepts ``highest``, ``high`` or ``medium``.
    When unset, this helper leaves PyTorch settings unchanged. CPU and MPS
    calls are no-ops. On CUDA it sets float32 matmul precision and enables
    cuDNN benchmarking when cuDNN is available. ``highest`` retains full
    float32 matmul precision; the other modes may use reduced-precision
    internal operations on supported hardware.

    cuDNN benchmarking chooses algorithms for observed input shapes and can
    add initial search time or affect reproducibility. This helper does not
    enable deterministic algorithms or guarantee identical seeded results.
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


def maybe_compile_network(network, device):
    """Wrap ``network`` with ``torch.compile`` when the env asks for it.

    Gated on ``MHCFLURRY_TORCH_COMPILE=1`` and a CUDA device.
    ``torch.compile`` is heavy: JIT graph capture + kernel fusion + on-
    disk cache + recompile-on-shape-change. The TF32 knob is cheaper
    and independent — see ``configure_matmul_precision``.

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
    resolve_torchinductor_compile_threads_env()
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


def maybe_compile_loss(loss_obj, device):
    """Wrap a loss module with ``torch.compile`` when the env asks for it.

    Gated on ``MHCFLURRY_TORCH_COMPILE=1`` and a CUDA device — same
    criteria as ``maybe_compile_network``. ``MSEWithInequalities``
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
    resolve_torchinductor_compile_threads_env()
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


def uncompiled_network(network):
    """Return the eager module behind ``network``."""
    return network._orig_mod if hasattr(network, "_orig_mod") else network


def validation_forward_network(network, eager_network):
    """Choose the module used for validation forward passes during training."""
    if os.environ.get("MHCFLURRY_TORCH_COMPILE_VALIDATION", "0") == "1":
        return network
    return eager_network


def effective_validation_batch_size(
        device,
        configured_batch_size,
        minibatch_size,
        model=None,
        num_workers_per_gpu=1,
        total_rows=None):
    """Return the validation batch size to use for the current device.

    CUDA validation uses the same live-memory budget as inference. Callers
    compute this once per fit and persist it in ``fit_info`` so torch.compile
    sees a stable shape across epochs. Explicit configured values are never
    changed.
    """
    if configured_batch_size:
        configured_batch_size = int(configured_batch_size)
        if configured_batch_size < 1:
            raise ValueError("validation batch size must be at least 1")
        return configured_batch_size
    device_type = getattr(device, "type", device)
    if device_type in ("cuda", "mps") and model is not None:
        from .pytorch_sizing import compute_prediction_batch_size
        return compute_prediction_batch_size(
            device,
            model=model,
            num_workers_per_gpu=num_workers_per_gpu,
            total_rows=total_rows,
        )
    if device_type == "cuda":
        # Parent-process planning cannot initialize CUDA. Keep a deterministic
        # analytic fallback until the worker has a live model and memory view.
        rows = max(4 * minibatch_size, 4096)
        return min(rows, int(total_rows)) if total_rows is not None else rows
    return 4 * minibatch_size
