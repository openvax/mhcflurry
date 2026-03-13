"""
Pytest configuration and session-wide initialization.
"""
from . import initialize

# Ensure deterministic test setup without per-file initialize() calls.
initialize()

def pytest_configure(config):
    # Register custom marks used across tests.
    config.addinivalue_line("markers", "slow: marks tests as slow")

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
