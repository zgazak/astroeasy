"""Astroeasy - Astrometry.net made easy.

A standalone Python package for plate-solving with astrometry.net.
Provides a clean, Pythonic API and easy Docker-based installation.

Example:
    >>> from astroeasy import (
    ...     AstrometryConfig,
    ...     Detection,
    ...     ImageMetadata,
    ...     solve_field,
    ... )
    >>> config = AstrometryConfig(
    ...     indices_path="/data/indices/5200-LITE",
    ...     docker_image="astrometry-cli",
    ... )
    >>> detections = [Detection(x=100, y=200, flux=1000), ...]
    >>> metadata = ImageMetadata(width=4096, height=4096)
    >>> result = solve_field(detections, metadata, config=config)
    >>> if result.success:
    ...     print(f"Solved! Center: {result.wcs.center_ra}, {result.wcs.center_dec}")
"""

from importlib.metadata import version

from astroeasy.config import AstrometryConfig
from astroeasy.constants import AstrometryIndexSeries
from astroeasy.indices import download_indices, examine_indices
from astroeasy.models import (
    Detection,
    ImageMetadata,
    MatchedStar,
    SolveResult,
    WCSResult,
    WCSStatus,
)
from astroeasy.runner import solve_field, test_install

__version__ = version("astroeasy")

__all__ = [
    # Configuration
    "AstrometryConfig",
    # Models
    "Detection",
    "ImageMetadata",
    "MatchedStar",
    "SolveResult",
    "WCSResult",
    "WCSStatus",
    # Constants
    "AstrometryIndexSeries",
    # Functions
    "solve_field",
    "test_install",
    "examine_indices",
    "download_indices",
]
