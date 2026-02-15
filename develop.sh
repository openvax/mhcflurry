#!/bin/bash
# Development environment setup script
# Source this script to activate the venv: source develop.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Check if already in the venv
if [[ "$VIRTUAL_ENV" == "$VENV_DIR" ]]; then
    echo "Virtual environment already active."
else
    if [[ -d "$VENV_DIR" ]]; then
        source "$VENV_DIR/bin/activate"
        echo "Activated virtual environment: $VENV_DIR"
    else
        echo "Virtual environment not found. Creating and installing..."
        python -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install -e .
        echo "Activated virtual environment: $VENV_DIR"
    fi
fi
