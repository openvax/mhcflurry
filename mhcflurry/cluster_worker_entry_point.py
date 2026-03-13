"""
Module entry point for cluster workers to ensure the current interpreter is used.
"""

from .cluster_parallelism import worker_entry_point


if __name__ == "__main__":
    worker_entry_point()
