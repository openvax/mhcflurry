"""Backward-compatible import for the historical histogram implementation.

New code should use the explicit HistogramPercentRankTransform name.
The alias also allows historical pickle references to resolve.
"""

from .histogram_percent_rank_transform import HistogramPercentRankTransform

PercentRankTransform = HistogramPercentRankTransform

__all__ = ["HistogramPercentRankTransform", "PercentRankTransform"]
