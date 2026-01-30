"""High-level solve_field API for astrometry.net plate solving."""

import logging

from astroeasy.config import AstrometryConfig
from astroeasy.dotnet.docker import test_dotnet_install as test_docker_install
from astroeasy.dotnet.local import test_dotnet_install as test_local_install
from astroeasy.dotnet.runner import solve_field as solve_field_dotnet
from astroeasy.models import (
    Detection,
    ImageMetadata,
    SolveResult,
    WCSResult,
    WCSStatus,
)

logger = logging.getLogger(__name__)


def solve_field(
    detections: list[Detection],
    metadata: ImageMetadata,
    config: AstrometryConfig,
    existing_wcs: WCSResult | None = None,
) -> SolveResult:
    """Solve astrometry for detected sources.

    This is the main entry point for plate solving. It accepts a list of
    detections with pixel coordinates and returns a WCS solution if successful.

    Args:
        detections: List of detected sources with x, y, and optional flux.
        metadata: Image metadata including width, height, and optional hints.
        config: Astrometry configuration including indices path and settings.
        existing_wcs: Optional existing WCS to verify/refine instead of blind solving.

    Returns:
        SolveResult containing:
        - success: Whether the solve was successful
        - status: Detailed status (SUCCESS, FAILED, INSUFFICIENT_SOURCES, TIMEOUT)
        - wcs: WCS solution if successful
        - matched_stars: Catalog stars matched to detections
        - detections: Original detections submitted
        - image_metadata: Image metadata used

    Example:
        >>> from astroeasy import AstrometryConfig, Detection, ImageMetadata, solve_field
        >>> config = AstrometryConfig(
        ...     indices_path="/data/indices/5200-LITE",
        ...     docker_image="astrometry-cli",
        ... )
        >>> detections = [Detection(x=100, y=200, flux=1000), ...]
        >>> metadata = ImageMetadata(width=4096, height=4096)
        >>> result = solve_field(detections, metadata, config)
        >>> if result.success:
        ...     print(f"Solved! Center: {result.wcs.center_ra}, {result.wcs.center_dec}")
    """
    logger.info(f"Attempting astrometric solution on {len(detections)} sources")

    # Check minimum sources
    if len(detections) < config.min_sources_for_attempt:
        logger.error(
            f"{len(detections)} sources (less than {config.min_sources_for_attempt}) "
            "found, skipping astrometry"
        )
        return SolveResult(
            success=False,
            status=WCSStatus.INSUFFICIENT_SOURCES,
            wcs=None,
            matched_stars=[],
            detections=detections,
            image_metadata=metadata,
        )

    # Dispatch to dotnet runner
    return solve_field_dotnet(detections, metadata, config, existing_wcs)


def test_install(docker_image: str | None = None) -> bool:
    """Test if astrometry.net is properly installed.

    Args:
        docker_image: Docker image name to test. If None, tests local installation.

    Returns:
        True if solve-field is available and working, False otherwise.
    """
    if docker_image is None:
        return test_local_install()
    else:
        return test_docker_install(docker_image)
