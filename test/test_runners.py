"""Offline tests for the mhcflurry.runners framework.

Covers the pieces we can test without actually provisioning cloud
resources: DSL rendering, config validation, instance-picker logic,
CLI arg parsing. Backends that drive real SSH/docker/modal are exercised
by `jobs/pan_allele_smoketest.py` against the three backends — that's
the closest thing we have to an integration test.
"""

import json
import sys
import types
from unittest import mock

import pytest

from mhcflurry.runners import App, BrevConfig, Image
from mhcflurry.runners._brev import (
    _brev_gpu_name,
    _pick_instance_type,
    _render_ops_script,
    _validate_config,
)


# ---- Image DSL rendering --------------------------------------------------

def _sample_image():
    return (
        Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
        .apt_install("bzip2", "rsync")
        .pip_install("pandas>=2.0", "scikit-learn")
        .pip_install_local_dir(".", editable=True)
    )


def test_image_from_registry_records_base():
    img = Image.from_registry("ubuntu:22.04")
    assert img.base == "ubuntu:22.04"
    assert img.dockerfile is None
    assert img.ops == ()


def test_image_layer_ops_chain_is_immutable():
    a = Image.from_registry("ubuntu:22.04")
    b = a.apt_install("curl")
    assert a.ops == ()
    assert len(b.ops) == 1
    assert b.ops[0].kind == "apt_install"
    assert b.ops[0].args == ("curl",)


def test_image_cant_layer_on_from_dockerfile():
    img = Image.from_dockerfile("docker/Dockerfile.train")
    with pytest.raises(ValueError, match="from_dockerfile"):
        img.pip_install("requests")


def test_render_dockerfile_uses_exec_form_for_pip():
    # Exec form is what prevents `pandas>=2.0` from being parsed as a
    # shell redirect. Regression test for that bug.
    df = _sample_image().render_dockerfile()
    assert 'RUN ["pip", "install", "--no-cache-dir", "pandas>=2.0"' in df
    assert "COPY . /workspace" in df
    assert 'RUN ["pip", "install", "--no-cache-dir", "-e", "/workspace"]' in df
    assert "FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime" in df


def test_render_dockerfile_requires_from_registry():
    img = Image.from_dockerfile("docker/Dockerfile.train")
    with pytest.raises(ValueError, match="from_registry"):
        img.render_dockerfile()


# ---- Brev ops-script rendering --------------------------------------------

def test_render_ops_script_uses_shlex_quote_for_packages():
    script = _render_ops_script(_sample_image())
    # Each op becomes a semicolon-joined bash line. Version specifiers
    # with shell metacharacters must be single-quoted by shlex.quote.
    assert "'pandas>=2.0'" in script
    assert "bzip2 rsync" in script
    assert "pip install --quiet -e \"$HOME/mhcflurry\"" in script


def test_render_ops_script_rejects_dockerfile_image():
    with pytest.raises(RuntimeError, match="from_dockerfile"):
        _render_ops_script(Image.from_dockerfile("docker/Dockerfile.train"))


# ---- BrevConfig validation ------------------------------------------------

def test_brev_validate_default_ok():
    _validate_config(BrevConfig())  # does not raise


def test_brev_validate_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        _validate_config(BrevConfig(mode="kubernetes"))


def test_brev_validate_container_plus_use_docker_false():
    with pytest.raises(ValueError, match="contradictory"):
        _validate_config(BrevConfig(mode="container", use_docker=False))


def test_brev_validate_vm_plus_use_docker_false_is_allowed():
    # Legacy escape hatch for mode=vm.
    _validate_config(BrevConfig(mode="vm", use_docker=False))


# ---- GPU label translation (Modal → Brev) ---------------------------------

@pytest.mark.parametrize("modal_label, expected_brev_name", [
    ("T4", "T4"),
    ("L4", "L4"),
    ("A100-40GB", "A100"),
    ("A100-80GB", "A100"),
    ("H100", "H100"),
    ("a100-40gb", "A100"),  # accept lowercased too
])
def test_brev_gpu_name_strips_vram_suffix(modal_label, expected_brev_name):
    assert _brev_gpu_name(modal_label) == expected_brev_name


# ---- Instance picker → `brev search` -------------------------------------

def _fn_with(gpu=None, min_cpu=None, min_memory=None,
             min_gpu_memory=None, min_disk=None):
    return types.SimpleNamespace(
        gpu=gpu, min_cpu=min_cpu, min_memory=min_memory,
        min_gpu_memory=min_gpu_memory, min_disk=min_disk,
    )


def test_pick_instance_type_builds_correct_search_cmd():
    captured = {}

    def fake_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        return mock.Mock(
            returncode=0,
            stdout=json.dumps([{"type": "n1-highmem-4:nvidia-tesla-t4:1"}]),
        )

    fn = _fn_with(gpu="T4", min_cpu=4, min_memory=26, min_gpu_memory=16,
                  min_disk=100)
    with mock.patch("mhcflurry.runners._brev.subprocess.run", fake_run):
        result = _pick_instance_type(fn)
    assert result == "n1-highmem-4:nvidia-tesla-t4:1"
    cmd = captured["cmd"]
    assert cmd[:3] == ["brev", "search", "gpu"]
    assert "--gpu-name" in cmd and "T4" in cmd
    assert "--min-vcpu" in cmd and "4" in cmd
    assert "--min-ram" in cmd and "26" in cmd
    assert "--min-vram" in cmd and "16" in cmd
    assert "--min-disk" in cmd and "100" in cmd


def test_pick_instance_type_cpu_when_no_gpu():
    def fake_run(cmd, *a, **kw):
        # Verify we called `search cpu` not `search gpu` when gpu=None.
        assert cmd[:3] == ["brev", "search", "cpu"]
        return mock.Mock(returncode=0, stdout=json.dumps([{"type": "n2d-highmem-2"}]))

    with mock.patch("mhcflurry.runners._brev.subprocess.run", fake_run):
        result = _pick_instance_type(_fn_with(min_memory=16))
    assert result == "n2d-highmem-2"


def test_pick_instance_type_returns_none_on_no_match():
    with mock.patch(
        "mhcflurry.runners._brev.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="[]"),
    ):
        assert _pick_instance_type(_fn_with(gpu="H100", min_memory=999999)) is None


# ---- App / @app.function --------------------------------------------------

def test_function_decorator_records_resource_requests():
    app = App("t")

    @app.function(
        image=_sample_image(),
        gpu="T4",
        min_cpu=4,
        min_memory=26,
        min_gpu_memory=16,
        min_disk=100,
        timeout=3600,
        env={"FOO": "bar"},
    )
    def train():
        pass

    assert train.gpu == "T4"
    assert train.min_cpu == 4
    assert train.min_memory == 26
    assert train.min_gpu_memory == 16
    assert train.min_disk == 100
    assert train.timeout == 3600
    assert train.env == {"FOO": "bar"}


def test_function_plain_call_raises_with_helpful_message():
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train():
        return "ran"

    with pytest.raises(RuntimeError, match="\\.local|\\.remote"):
        train()
    assert train.local() == "ran"  # local() still works


def test_remote_requires_backend_to_be_selected():
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train():
        pass

    with pytest.raises(RuntimeError, match="No backend selected"):
        train.remote()


def test_remote_rejects_non_json_args():
    app = App("t")

    @app.function(image=_sample_image(), gpu="T4")
    def train(x):
        return x

    app._backend = "local"
    with pytest.raises(TypeError, match="JSON-serializable"):
        train.remote(object())


# ---- CLI entry ------------------------------------------------------------

def test_cli_errors_without_instance_on_brev():
    from mhcflurry.runners import _cli

    with mock.patch.object(sys, "argv",
                           ["mhcflurry-run", "brev", "jobs/pan_allele_dsl.py"]):
        with pytest.raises(SystemExit):
            _cli.main(["brev", "jobs/pan_allele_dsl.py"])
