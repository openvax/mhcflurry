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
Pytest configuration and session-wide initialization.
"""
import pytest

from . import initialize

# Ensure deterministic test setup without per-file initialize() calls.
initialize()

from mhcflurry.common import configure_pytorch  # noqa: E402

# Unit tests default to CPU for speed and determinism. Tests that exercise
# CUDA/MPS opt into those backends explicitly.
configure_pytorch(backend="cpu")


@pytest.fixture(autouse=True)
def _default_pytorch_backend_cpu():
    configure_pytorch(backend="cpu")
    yield
    configure_pytorch(backend="cpu")


@pytest.fixture(scope="session")
def released_allele_specific_predictor():
    from mhcflurry import Class1AffinityPredictor
    from mhcflurry.downloads import get_path
    return Class1AffinityPredictor.load(get_path("models_class1", "models"))


@pytest.fixture(scope="session")
def released_pan_allele_predictor():
    from mhcflurry import Class1AffinityPredictor
    from mhcflurry.downloads import get_path
    return Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"))


@pytest.fixture(scope="session")
def released_pan_allele_predictor_two_models():
    from mhcflurry import Class1AffinityPredictor
    from mhcflurry.downloads import get_path
    return Class1AffinityPredictor.load(
        get_path("models_class1_pan", "models.combined"),
        max_models=2)


@pytest.fixture(scope="session")
def released_affinity_predictors(
        released_allele_specific_predictor,
        released_pan_allele_predictor):
    return {
        "allele-specific": released_allele_specific_predictor,
        "pan-allele": released_pan_allele_predictor,
    }


@pytest.fixture(scope="session")
def released_affinity_predictors_two_pan_models(
        released_allele_specific_predictor,
        released_pan_allele_predictor_two_models):
    return {
        "allele-specific": released_allele_specific_predictor,
        "pan-allele": released_pan_allele_predictor_two_models,
    }


def pytest_configure(config):
    # Register custom marks used across tests.
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line(
        "markers",
        "integration: marks end-to-end command/training tests",
    )
    config.addinivalue_line(
        "markers",
        "downloads: marks tests that require locally cached MHCflurry bundles",
    )

    # PyTorch warns that padding='same' with even kernels may allocate a
    # temporary padded copy. This is expected for our processing defaults.
    config.addinivalue_line(
        "filterwarnings",
        (
            "ignore:Using padding='same' with even kernel lengths and odd "
            "dilation may require a zero-padded copy of the input be created.*:"
            "UserWarning:torch\\.nn\\.modules\\.conv"
        ),
    )
