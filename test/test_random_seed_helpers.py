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

"""Tests for the CLI-level random-seed reproducibility helpers.

Covers the helper surface that makes mhcflurry runs reproducible out of the
box: ``DEFAULT_RANDOM_SEED``, ``add_random_seed_arg``, ``configure_random_seed``,
and ``derive_seed`` (all in ``mhcflurry.common``). These are unit-level checks
on the helpers themselves — the slow end-to-end "same seed -> same model"
training path is exercised in the integration suites, not here.
"""

import argparse
import os
import random
import subprocess
import sys

import numpy

from mhcflurry.common import (
    DEFAULT_RANDOM_SEED,
    add_random_seed_arg,
    configure_random_seed,
    derive_seed,
)


def test_default_random_seed_is_42():
    assert DEFAULT_RANDOM_SEED == 42


def test_add_random_seed_arg_default():
    parser = argparse.ArgumentParser()
    add_random_seed_arg(parser)
    args = parser.parse_args([])
    assert args.random_seed == DEFAULT_RANDOM_SEED

    # And an explicit value is honored / parsed as an int.
    args = parser.parse_args(["--random-seed", "123"])
    assert args.random_seed == 123


def _draw_sequence():
    """Draw from each global RNG that configure_random_seed seeds."""
    import torch
    return (
        [random.random() for _ in range(5)],
        numpy.random.rand(5).tolist(),
        torch.rand(5).tolist(),
    )


def test_configure_random_seed_is_deterministic():
    resolved_a = configure_random_seed(7)
    py_a, np_a, torch_a = _draw_sequence()

    resolved_b = configure_random_seed(7)
    py_b, np_b, torch_b = _draw_sequence()

    # configure_random_seed returns the resolved seed.
    assert resolved_a == 7
    assert resolved_b == 7

    # Same seed -> identical draws across all three RNGs.
    assert py_a == py_b
    assert np_a == np_b
    assert torch_a == torch_b


def test_configure_random_seed_differs_across_seeds():
    configure_random_seed(7)
    py_a, np_a, torch_a = _draw_sequence()

    configure_random_seed(8)
    py_b, np_b, torch_b = _draw_sequence()

    assert py_a != py_b
    assert np_a != np_b
    assert torch_a != torch_b


def test_configure_random_seed_none_draws_usable_seed():
    # None means "draw from entropy": no determinism, but a usable int seed
    # must be returned (callers derive sub-seeds from it).
    resolved = configure_random_seed(None)
    assert isinstance(resolved, int)
    assert 0 <= resolved < 2 ** 32

    # Two entropy draws should (with overwhelming probability) differ.
    other = configure_random_seed(None)
    assert resolved != other


def test_derive_seed_is_deterministic():
    a = derive_seed(42, "fit", 0, 0, 0)
    b = derive_seed(42, "fit", 0, 0, 0)
    assert a == b


def test_derive_seed_distinct_for_distinct_coords():
    coords = [
        ("fit", 0, 0, 0),
        ("fit", 0, 0, 1),
        ("fit", 0, 1, 0),
        ("fit", 1, 0, 0),
        ("fit", 0, 0),
        ("calibrate", 0, 0, 0),
    ]
    seeds = [derive_seed(42, *c) for c in coords]
    assert len(set(seeds)) == len(seeds), "sub-seeds collided: %r" % seeds

    # A different master seed also changes the derived value.
    assert derive_seed(42, "fit", 0, 0, 0) != derive_seed(43, "fit", 0, 0, 0)


def test_derive_seed_is_non_negative_int():
    # Downstream truncates with `% (2**32)` for numpy; the raw value must be a
    # non-negative int so that truncation is well-defined.
    for coords in [("fit", 0, 0, 0), ("calibrate", 5, 9), ("x",)]:
        seed = derive_seed(42, *coords)
        assert isinstance(seed, int)
        assert seed >= 0
        assert seed % (2 ** 32) >= 0


def test_derive_seed_handles_none_master():
    # A legacy run with no recorded master seed still derives stably.
    a = derive_seed(None, "fit", 0)
    b = derive_seed(None, "fit", 0)
    assert a == b
    assert isinstance(a, int) and a >= 0


def test_derive_seed_stable_across_processes():
    # derive_seed must use stable hashing (hashlib), NOT Python's salted
    # builtin hash(), so a sub-seed is identical across separate process
    # invocations (e.g. --only-initialize / --continue-incomplete split).
    expected = derive_seed(42, "fit", 0, 0, 0)
    code = (
        "from mhcflurry.common import derive_seed; "
        "print(derive_seed(42, 'fit', 0, 0, 0))")
    env = dict(os.environ, PYTHONHASHSEED="random")
    out = subprocess.check_output([sys.executable, "-c", code], env=env)
    assert int(out.strip()) == expected
