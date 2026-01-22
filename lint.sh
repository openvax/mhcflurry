#!/bin/bash
set -o errexit

# Lint using ruff (fast Python linter)
# Run from project root directory

echo "Running ruff linter..."
ruff check mhcflurry/ test/ --output-format=concise

echo "Passes ruff check"
