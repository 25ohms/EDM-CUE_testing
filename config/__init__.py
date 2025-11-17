"""
Configuration package for EDM-CUE visualizations.

Exports taxonomy-related settings that are shared across modules.
"""

from .taxonomy import (
    BPM_BUCKETS,
    GENRE_FONT_STACK,
    GENRE_TAXONOMY,
    GENRE_DELIMITER_PATTERN,
    GENRE_SCORING,
)

__all__ = [
    "BPM_BUCKETS",
    "GENRE_FONT_STACK",
    "GENRE_TAXONOMY",
    "GENRE_DELIMITER_PATTERN",
    "GENRE_SCORING",
]
