#!/bin/bash
# Development environment setup script
# Source this script to activate the venv: source develop.sh

if [ -n "${ZSH_VERSION:-}" ]; then
    _MHCFLURRY_SCRIPT_PATH="${(%):-%x}"
elif [ -n "${BASH_VERSION:-}" ]; then
    _MHCFLURRY_SCRIPT_PATH="${BASH_SOURCE[0]}"
else
    _MHCFLURRY_SCRIPT_PATH="$0"
fi

SCRIPT_DIR="$(cd "$(dirname "$_MHCFLURRY_SCRIPT_PATH")" && pwd)"
unset _MHCFLURRY_SCRIPT_PATH

if [[ ! -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    echo "Error: could not locate mhcflurry repo root (expected $SCRIPT_DIR/pyproject.toml)." >&2
    echo "Source this script with bash or zsh: source path/to/mhcflurry/develop.sh" >&2
    return 1 2>/dev/null || exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"

# Check if already in the venv
if [[ "${VIRTUAL_ENV:-}" == "$VENV_DIR" ]]; then
    echo "Virtual environment already active."
else
    if [[ -d "$VENV_DIR" ]]; then
        source "$VENV_DIR/bin/activate"
        echo "Activated virtual environment: $VENV_DIR"
    else
        echo "Virtual environment not found. Creating and installing..."
        _MHCFLURRY_PYTHON="$(command -v python3 || command -v python)"
        if [[ -z "$_MHCFLURRY_PYTHON" ]]; then
            echo "Error: python3 not found on PATH." >&2
            unset _MHCFLURRY_PYTHON
            return 1 2>/dev/null || exit 1
        fi
        "$_MHCFLURRY_PYTHON" -m venv "$VENV_DIR"
        unset _MHCFLURRY_PYTHON
        source "$VENV_DIR/bin/activate"
        pip install -e "$SCRIPT_DIR"
        echo "Activated virtual environment: $VENV_DIR"
    fi
fi
