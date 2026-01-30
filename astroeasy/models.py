"""Data models for astrometry.net plate solving."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Detection:
    """A detected source in image coordinates.

    Attributes:
        x: X pixel coordinate.
        y: Y pixel coordinate.
        flux: Source flux/brightness (optional).
    """

    x: float
    y: float
    flux: float | None = None


@dataclass
class ImageMetadata:
    """Metadata about the image being solved.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        boresight_ra: Approximate RA of image center in degrees (optional hint).
        boresight_dec: Approximate Dec of image center in degrees (optional hint).
    """

    width: int
    height: int
    boresight_ra: float | None = None
    boresight_dec: float | None = None


class WCSStatus(str, Enum):
    """Status of a WCS solution attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    INSUFFICIENT_SOURCES = "insufficient_sources"
    TIMEOUT = "timeout"


@dataclass
class WCSResult:
    """Hybrid WCS representation - raw data + conversion methods.

    Stores both the raw FITS header dictionary (preserving exact astrometry.net output)
    and provides convenient access to core WCS parameters.

    Attributes:
        raw_header: Raw FITS header dictionary from astrometry.net.
        crval1: RA at reference pixel (degrees).
        crval2: Dec at reference pixel (degrees).
        crpix1: Reference pixel X coordinate.
        crpix2: Reference pixel Y coordinate.
        cd_matrix: CD matrix elements (CD1_1, CD1_2, CD2_1, CD2_2).
    """

    raw_header: dict[str, Any]
    crval1: float
    crval2: float
    crpix1: float
    crpix2: float
    cd_matrix: tuple[float, float, float, float]

    @classmethod
    def from_fits_header(cls, header) -> "WCSResult":
        """Create WCSResult from a FITS header.

        Args:
            header: FITS header object or dictionary with WCS keywords.

        Returns:
            WCSResult instance.
        """
        # Convert header to dictionary if it's a FITS Header object
        if hasattr(header, "tostring"):
            raw_header = dict(header)
        else:
            raw_header = dict(header)

        return cls(
            raw_header=raw_header,
            crval1=float(header.get("CRVAL1", 0.0)),
            crval2=float(header.get("CRVAL2", 0.0)),
            crpix1=float(header.get("CRPIX1", 0.0)),
            crpix2=float(header.get("CRPIX2", 0.0)),
            cd_matrix=(
                float(header.get("CD1_1", 0.0)),
                float(header.get("CD1_2", 0.0)),
                float(header.get("CD2_1", 0.0)),
                float(header.get("CD2_2", 0.0)),
            ),
        )

    def to_fits_header(self) -> dict[str, Any]:
        """Get raw FITS header dictionary.

        Returns:
            Dictionary of FITS header keywords and values.
        """
        return self.raw_header

    def to_astropy_wcs(self):
        """Convert to astropy.wcs.WCS object.

        Returns:
            astropy.wcs.WCS object.
        """
        from astropy.io import fits
        from astropy.wcs import WCS

        header = fits.Header()
        for key, value in self.raw_header.items():
            if key and not key.startswith("COMMENT") and not key.startswith("HISTORY"):
                try:
                    header[key] = value
                except (ValueError, KeyError):
                    # Skip keys that can't be added to a FITS header
                    pass
        return WCS(header)

    @property
    def center_ra(self) -> float:
        """RA at the reference pixel (degrees)."""
        return self.crval1

    @property
    def center_dec(self) -> float:
        """Dec at the reference pixel (degrees)."""
        return self.crval2

    @property
    def pixel_scale(self) -> float:
        """Approximate pixel scale in arcseconds per pixel."""
        import math

        # Calculate from CD matrix
        cd1_1, cd1_2, cd2_1, cd2_2 = self.cd_matrix
        # Pixel scale is sqrt of determinant of CD matrix, converted to arcsec
        scale = math.sqrt(abs(cd1_1 * cd2_2 - cd1_2 * cd2_1)) * 3600.0
        return scale


@dataclass
class MatchedStar:
    """A star from the reference catalog that was matched to a detection.

    Attributes:
        x: X pixel coordinate.
        y: Y pixel coordinate.
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        magnitude: Star magnitude (optional).
        catalog: Reference catalog name (optional).
        catalog_id: Star ID in the reference catalog (optional).
    """

    x: float
    y: float
    ra: float
    dec: float
    magnitude: float | None = None
    catalog: str | None = None
    catalog_id: str | None = None


@dataclass
class SolveResult:
    """Result of a plate-solving attempt.

    Attributes:
        success: Whether the solve was successful.
        status: Detailed status of the solve attempt.
        wcs: WCS solution if successful, None otherwise.
        matched_stars: List of catalog stars matched to detections.
        detections: Original detections that were submitted for solving.
        image_metadata: Metadata about the solved image.
    """

    success: bool
    status: WCSStatus
    wcs: WCSResult | None
    matched_stars: list[MatchedStar]
    detections: list[Detection]
    image_metadata: ImageMetadata
