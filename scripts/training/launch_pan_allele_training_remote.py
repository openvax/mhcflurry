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

"""
Launch pan-allele training on remote GPU machines.

This is the maintained runplz wrapper for pan_allele_release_full.sh. The
release workflow chooses whether Brev should use an existing instance or
intentionally provision one by setting RUNPLZ_BREV_* environment variables
before invoking ``runplz brev``. Local runs should call the shell script
directly.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

try:
    from runplz import App, Image
    from runplz.config import BrevConfig
except ImportError as e:
    raise SystemExit(
        "runplz is required for this launcher. Install it with "
        "`pip install runplz` or run pan_allele_release_full.sh locally."
    ) from e


APP_NAME = os.environ.get("RUNPLZ_APP_NAME", "mhcflurry-pan-allele-training")
GPU_TYPE = os.environ.get("RUNPLZ_GPU", "A100")
NUM_GPUS = int(os.environ.get("RUNPLZ_NUM_GPUS", "4"))
MIN_GPU_MEMORY = int(os.environ.get("RUNPLZ_MIN_GPU_MEMORY", "35"))
MIN_CPU = int(os.environ.get("RUNPLZ_MIN_CPU", "32"))
MIN_MEMORY = int(os.environ.get("RUNPLZ_MIN_MEMORY", "300"))
MIN_DISK = int(os.environ.get("RUNPLZ_MIN_DISK", "1000"))
DEFAULT_OUT = os.environ.get(
    "MHCFLURRY_OUT", "/root/mhcflurry-pan-allele-training-run"
)

TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def env_bool(environ, name, default=False):
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_ENV_VALUES:
        return True
    if normalized in FALSE_ENV_VALUES:
        return False
    raise ValueError(
        "%s must be one of %s or %s; got %r" % (
            name,
            sorted(TRUE_ENV_VALUES),
            sorted(FALSE_ENV_VALUES),
            value,
        )
    )


def compare_torch_compile_value(environ):
    value = environ.get(
        "COMPARE_TORCH_COMPILE",
        environ.get("MHCFLURRY_TORCH_COMPILE", "auto"),
    )
    normalized = value.strip().lower()
    if normalized in TRUE_ENV_VALUES:
        return "1"
    if normalized in FALSE_ENV_VALUES:
        return "0"
    if normalized == "auto":
        return "auto"
    raise ValueError(
        "COMPARE_TORCH_COMPILE must be auto, true/false, or 1/0; got %r" %
        value
    )


def compare_matmul_precision_value(environ):
    value = environ.get(
        "COMPARE_MATMUL_PRECISION",
        environ.get("MHCFLURRY_MATMUL_PRECISION", "high"),
    )
    normalized = value.strip().lower()
    if normalized not in {"none", "highest", "high", "medium"}:
        raise ValueError(
            "COMPARE_MATMUL_PRECISION must be one of none, highest, high, "
            "medium; got %r" % value
        )
    return normalized


def env_int_optional(environ, name):
    value = environ.get(name)
    if value is None or not value.strip():
        return None
    return int(value)


def env_csv_tuple(environ, name, default):
    value = environ.get(name)
    if value is None:
        return default
    if not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def brev_config_from_env(environ=os.environ):
    return BrevConfig(
        auto_create_instances=env_bool(
            environ, "RUNPLZ_BREV_AUTO_CREATE", default=False
        ),
        instance_type=(
            environ.get("RUNPLZ_BREV_INSTANCE_TYPE")
            or environ.get("BREV_INSTANCE_TYPE")
            or None
        ),
        mode=environ.get("RUNPLZ_BREV_MODE", "container"),
        on_finish=environ.get("RUNPLZ_BREV_ON_FINISH", "leave"),
        max_runtime_seconds=env_int_optional(
            environ, "RUNPLZ_BREV_MAX_RUNTIME_SECONDS"
        ),
        ssh_ready_wait_seconds=int(
            environ.get("RUNPLZ_BREV_SSH_READY_WAIT_SECONDS", "2400")
        ),
        instance_type_fallback_count=int(
            environ.get("RUNPLZ_BREV_INSTANCE_TYPE_FALLBACK_COUNT", "3")
        ),
        exclude_providers=env_csv_tuple(
            environ, "RUNPLZ_BREV_EXCLUDE_PROVIDERS", ("oci",)
        ),
    )


def remote_training_env(environ=os.environ):
    env = {
        "COMPARE_INCLUDE": environ.get(
            "COMPARE_INCLUDE", "affinity,processing,presentation"
        ),
        "COMPARE_BASELINE": environ.get("COMPARE_BASELINE", "public:2.0.0"),
        "COMPARE_BASELINE_LABEL": environ.get(
            "COMPARE_BASELINE_LABEL", "MHCflurry 2.0"
        ),
        "COMPARE_MAX_TASKS_PER_WORKER": environ.get(
            "COMPARE_MAX_TASKS_PER_WORKER",
            environ.get("MAX_TASKS_PER_WORKER", "12"),
        ),
        "COMPARE_MAX_WORKERS_PER_GPU": environ.get(
            "COMPARE_MAX_WORKERS_PER_GPU", "auto"
        ),
        "COMPARE_NUM_JOBS": environ.get("COMPARE_NUM_JOBS", "auto"),
        "COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER": environ.get(
            "COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER", "1"
        ),
        "COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU": environ.get(
            "COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU", "1"
        ),
        "COMPARE_PRESENTATION_NUM_JOBS": environ.get(
            "COMPARE_PRESENTATION_NUM_JOBS", "1"
        ),
        "COMPARE_PRESENTATION_TORCH_COMPILE": environ.get(
            "COMPARE_PRESENTATION_TORCH_COMPILE", "0"
        ),
        "DATALOADER_NUM_WORKERS": environ.get("DATALOADER_NUM_WORKERS", "auto"),
        "MAX_TASKS_PER_WORKER": environ.get("MAX_TASKS_PER_WORKER", "12"),
        "MAX_WORKERS_PER_GPU": environ.get("MAX_WORKERS_PER_GPU", "auto"),
        # PyTorch/Inductor workers load GNU OpenMP (libgomp). The PyTorch
        # conda image also includes mkl-service, whose INTEL threading default
        # aborts when libgomp is already loaded.
        "MKL_THREADING_LAYER": environ.get("MKL_THREADING_LAYER", "GNU"),
        "MHCFLURRY_ENABLE_TIMING": environ.get("MHCFLURRY_ENABLE_TIMING", "1"),
        "MHCFLURRY_TORCH_COMPILE": environ.get("MHCFLURRY_TORCH_COMPILE", "1"),
        "MHCFLURRY_TORCH_COMPILE_LOSS": environ.get(
            "MHCFLURRY_TORCH_COMPILE_LOSS", "1"
        ),
        "MHCFLURRY_MATMUL_PRECISION": environ.get(
            "MHCFLURRY_MATMUL_PRECISION", "high"
        ),
        "MATMUL_PRECISION": environ.get("MATMUL_PRECISION", "high"),
        "MATMUL_PRECISION_CLI": environ.get("MATMUL_PRECISION_CLI", "high"),
        "PRESENTATION_PROCESSING_WITH_FLANKS_KIND": environ.get(
            "PRESENTATION_PROCESSING_WITH_FLANKS_KIND", "with_flanks"
        ),
        "PRESENTATION_MODES": environ.get(
            "PRESENTATION_MODES", "with_flanks,without_flanks"
        ),
        "PAPER_FIGURES_SCORES_DIR": environ.get(
            "PAPER_FIGURES_SCORES_DIR",
            environ.get("PAPER_FIGURES_ARTIFACTS_DIR", ""),
        ),
        "PAPER_FIGURES_ARTIFACTS_DIR": environ.get(
            "PAPER_FIGURES_ARTIFACTS_DIR", ""
        ),
        "PAPER_FIGURES_MULTIALLELIC_PREDICTIONS": environ.get(
            "PAPER_FIGURES_MULTIALLELIC_PREDICTIONS", ""
        ),
        "PAPER_FIGURES_MONOALLELIC_PREDICTIONS": environ.get(
            "PAPER_FIGURES_MONOALLELIC_PREDICTIONS", ""
        ),
        "PAPER_FIGURES_FORMATS": environ.get(
            "PAPER_FIGURES_FORMATS", "svg,pdf,png"
        ),
        "PAPER_FIGURES_CANDIDATE_PREDICTOR": environ.get(
            "PAPER_FIGURES_CANDIDATE_PREDICTOR", ""
        ),
        "PAPER_FIGURES_EXTERNAL_BASELINES": environ.get(
            "PAPER_FIGURES_EXTERNAL_BASELINES", ""
        ),
        "PAPER_FIGURES_PREFERRED_PREDICTORS": environ.get(
            "PAPER_FIGURES_PREFERRED_PREDICTORS", ""
        ),
        "PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS": environ.get(
            "PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS", ""
        ),
        "PAPER_FIGURES_PRESENTATION_PANEL_BASELINES": environ.get(
            "PAPER_FIGURES_PRESENTATION_PANEL_BASELINES", ""
        ),
        "PROCESSING_VARIANTS": environ.get(
            "PROCESSING_VARIANTS", "with_flanks no_flank short_flanks"
        ),
        "PROCESSING_MODES": environ.get(
            "PROCESSING_MODES", "with_flanks,no_flank,short_flanks"
        ),
        "RUN_LABEL": environ.get("RUN_LABEL", "new"),
        "RUN_RELEASE_EVAL": environ.get("RUN_RELEASE_EVAL", "0"),
        "RUN_RELEASE_PLOTS": environ.get("RUN_RELEASE_PLOTS", "0"),
        "TORCHINDUCTOR_COMPILE_THREADS": environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS", "auto"
        ),
        "TRAINING_MINIBATCH_SIZE": environ.get("TRAINING_MINIBATCH_SIZE", "1024"),
    }
    for name in (
        "AFFINITY_MINIBATCH_SIZE",
        "AFFINITY_MAX_WORKERS_PER_GPU",
        "PROCESSING_MINIBATCH_SIZE",
    ):
        if name in environ:
            env[name] = environ[name]
    return env


image = (
    Image.from_registry(
        os.environ.get(
            "RUNPLZ_IMAGE",
            "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        )
    )
    .apt_install(
        "python-is-python3",
        "bzip2",
        "wget",
        "rsync",
        "build-essential",
        "git",
        "libhdf5-dev",
        "libxml2-dev",
        "libxslt1-dev",
        "procps",
    )
    .pip_install("matplotlib", "pypdf")
    .pip_install("runplz>=3.11.0")
    .pip_install_local_dir(".", editable=True)
)

app = App(
    APP_NAME,
    brev_config=brev_config_from_env(),
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    num_gpus=NUM_GPUS,
    min_gpu_memory=MIN_GPU_MEMORY,
    min_cpu=MIN_CPU,
    min_memory=MIN_MEMORY,
    min_disk=MIN_DISK,
    timeout=60 * 60 * 24 * 14,
    env=remote_training_env(),
)
def train_release_full():
    """Run the maintained full release training script."""
    repo = Path.cwd()
    out = Path(
        os.environ.get("RUNPLZ_OUT")
        or os.environ.get("MHCFLURRY_OUT")
        or DEFAULT_OUT
    ).resolve()
    env = os.environ.copy()
    env.update({"MHCFLURRY_OUT": str(out), "REPO": str(repo)})
    subprocess.run(
        ["bash", "scripts/training/pan_allele_release_full.sh"],
        check=True,
        cwd=repo,
        env=env,
    )
    if env_bool(env, "RUN_RELEASE_EVAL", default=False):
        run_release_evaluation(repo, out, env)
    if env_bool(env, "RUN_RELEASE_PLOTS", default=False):
        run_release_plots(repo, out, env)


def run_release_evaluation(repo, out, env):
    """Fetch release eval data and run compare-models on the remote GPU box."""
    eval_out = out / "eval_comparison"
    eval_out.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "mhcflurry",
            "downloads",
            "fetch",
            "data_evaluation",
            "models_class1_pan",
            "models_class1_processing",
            "models_class1_presentation",
        ],
        check=True,
        cwd=repo,
        env=env,
    )
    data_dir = subprocess.check_output(
        ["mhcflurry", "downloads", "path", "data_evaluation"],
        cwd=repo,
        env=env,
        text=True,
    ).strip()
    baseline = env.get("COMPARE_BASELINE", "public:2.0.0")
    if baseline.startswith("public:"):
        baseline_env = env.copy()
        baseline_env["MHCFLURRY_DOWNLOADS_CURRENT_RELEASE"] = (
            baseline.split(":", 1)[1]
        )
        subprocess.run(
            [
                "mhcflurry",
                "downloads",
                "fetch",
                "models_class1_pan",
                "models_class1_processing",
                "models_class1_presentation",
            ],
            check=True,
            cwd=repo,
            env=baseline_env,
        )

    compare_args = [
        "mhcflurry",
        "eval",
        "compare-models",
        "--a", str(out),
        "--a-label", env.get("RUN_LABEL", "new"),
        "--b", env.get("COMPARE_BASELINE", "public:2.0.0"),
        "--b-label", env.get("COMPARE_BASELINE_LABEL", "MHCflurry 2.0"),
        "--data-dir", data_dir,
        "--include", env.get("COMPARE_INCLUDE", "affinity,processing,presentation"),
        "--processing-modes", env.get(
            "PROCESSING_MODES", "with_flanks,no_flank,short_flanks"
        ),
        "--presentation-modes", env.get(
            "PRESENTATION_MODES", "with_flanks,without_flanks"
        ),
        "--out", str(eval_out),
        "--backend", env.get("COMPARE_BACKEND", "auto"),
        "--num-jobs", env.get("COMPARE_NUM_JOBS", "auto"),
        "--max-workers-per-gpu", env.get("COMPARE_MAX_WORKERS_PER_GPU", "auto"),
        "--max-tasks-per-worker", env.get("COMPARE_MAX_TASKS_PER_WORKER", "12"),
        "--presentation-num-jobs", env.get("COMPARE_PRESENTATION_NUM_JOBS", "1"),
        "--presentation-max-workers-per-gpu", env.get(
            "COMPARE_PRESENTATION_MAX_WORKERS_PER_GPU", "1"
        ),
        "--presentation-max-tasks-per-worker", env.get(
            "COMPARE_PRESENTATION_MAX_TASKS_PER_WORKER", "1"
        ),
        "--presentation-torch-compile", compare_torch_compile_value({
            "COMPARE_TORCH_COMPILE": env.get(
                "COMPARE_PRESENTATION_TORCH_COMPILE", "0"
            )
        }),
        "--worker-log-dir", str(eval_out / "worker_logs"),
        "--torch-compile", compare_torch_compile_value(env),
        "--matmul-precision", compare_matmul_precision_value(env),
    ]
    compare_gpus = env.get("COMPARE_GPUS", str(NUM_GPUS))
    if compare_gpus.strip().lower() != "auto":
        compare_args.extend(["--gpus", compare_gpus])
    subprocess.run(compare_args, check=True, cwd=repo, env=env)


def run_release_plots(repo, out, env):
    """Render compare-models plots before the remote instance is cleaned up."""
    plot_args = [
        "mhcflurry",
        "eval",
        "plot-comparison",
        "--input", str(out / "eval_comparison"),
        "--a-label", env.get("RUN_LABEL", "new"),
        "--b-label", env.get("COMPARE_BASELINE_LABEL", "MHCflurry 2.0"),
        "--summary-pdf",
        str(out / "eval_comparison" / "plots" / "model_comparison_figures.pdf"),
    ]
    scores_dir = env.get("PAPER_FIGURES_SCORES_DIR", "").strip()
    multiallelic_predictions = env.get(
        "PAPER_FIGURES_MULTIALLELIC_PREDICTIONS", "").strip()
    monoallelic_predictions = env.get(
        "PAPER_FIGURES_MONOALLELIC_PREDICTIONS", "").strip()
    if scores_dir or multiallelic_predictions or monoallelic_predictions:
        plot_args.extend([
            "--paper-figures-out",
            str(out / "eval_comparison" / "plots" / "paper_figures"),
            "--paper-figures-formats",
            env.get("PAPER_FIGURES_FORMATS", "svg,pdf,png"),
        ])
        if scores_dir:
            plot_args.extend(["--paper-figures-scores-dir", scores_dir])
        if multiallelic_predictions:
            plot_args.extend([
                "--paper-figures-multiallelic-predictions",
                multiallelic_predictions,
            ])
        if monoallelic_predictions:
            plot_args.extend([
                "--paper-figures-monoallelic-predictions",
                monoallelic_predictions,
            ])
        passthrough = (
            ("PAPER_FIGURES_CANDIDATE_PREDICTOR",
             "--paper-figures-candidate-predictor"),
            ("PAPER_FIGURES_EXTERNAL_BASELINES",
             "--paper-figures-external-baselines"),
            ("PAPER_FIGURES_PREFERRED_PREDICTORS",
             "--paper-figures-preferred-predictors"),
            ("PAPER_FIGURES_PRESENTATION_PANEL_PREDICTORS",
             "--paper-figures-presentation-panel-predictors"),
            ("PAPER_FIGURES_PRESENTATION_PANEL_BASELINES",
             "--paper-figures-presentation-panel-baselines"),
        )
        for env_name, flag in passthrough:
            value = env.get(env_name, "").strip()
            if value:
                plot_args.extend([flag, value])
    subprocess.run(plot_args, check=True, cwd=repo, env=env)


@app.local_entrypoint()
def main():
    train_release_full.remote()
