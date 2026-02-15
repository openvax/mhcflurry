"""
Pytest configuration and session-wide initialization.
"""
from . import initialize

# Ensure deterministic test setup without per-file initialize() calls.
initialize()
