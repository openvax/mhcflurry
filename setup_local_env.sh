#!/bin/bash
set -e

VENV_DIR=".venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate and install
echo "Activating virtual environment and installing mhcflurry..."
source "$VENV_DIR/bin/activate"
pip install -e .

echo ""
echo "Done! To use this environment, run:"
echo "  source .venv/bin/activate"
