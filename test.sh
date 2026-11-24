#!/bin/bash
set -o errexit

# Run the test suite via ``python -m pytest`` rather than the generated
# ``pytest`` console script so PyTorch can see MPS accelerators on macOS.
# Extra arguments are forwarded to pytest; when none are given, runs all tests
# under ``test/``.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="$(command -v python || command -v python3)"
fi

if [[ $# -eq 0 ]]; then
    exec "$PYTHON" -m pytest "$SCRIPT_DIR/test"
else
    exec "$PYTHON" -m pytest "$@"
fi
