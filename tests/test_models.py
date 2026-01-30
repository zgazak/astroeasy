"""Tests for astroeasy data models."""

import pytest

from astroeasy import Detection, ImageMetadata, MatchedStar, SolveResult, WCSResult, WCSStatus


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_create_with_flux(self):
        """Test creating a detection with flux."""
        det = Detection(x=100.5, y=200.3, flux=1000.0)
        assert det.x == 100.5
        assert det.y == 200.3
        assert det.flux == 1000.0

    def test_create_without_flux(self):
        """Test creating a detection without flux."""
        det = Detection(x=100.5, y=200.3)
        assert det.x == 100.5
        assert det.y == 200.3
        assert det.flux is None

    def test_equality(self):
        """Test detection equality."""
        det1 = Detection(x=100.0, y=200.0, flux=1000.0)
        det2 = Detection(x=100.0, y=200.0, flux=1000.0)
        assert det1 == det2

    def test_inequality(self):
        """Test detection inequality."""
        det1 = Detection(x=100.0, y=200.0, flux=1000.0)
        det2 = Detection(x=100.0, y=200.0, flux=2000.0)
        assert det1 != det2


class TestImageMetadata:
    """Tests for the ImageMetadata dataclass."""

    def test_create_minimal(self):
        """Test creating metadata with just width and height."""
        meta = ImageMetadata(width=1024, height=768)
        assert meta.width == 1024
        assert meta.height == 768
        assert meta.boresight_ra is None
        assert meta.boresight_dec is None

    def test_create_with_boresight(self):
        """Test creating metadata with boresight hints."""
        meta = ImageMetadata(
            width=4096,
            height=4096,
            boresight_ra=180.0,
            boresight_dec=45.0,
        )
        assert meta.width == 4096
        assert meta.height == 4096
        assert meta.boresight_ra == 180.0
        assert meta.boresight_dec == 45.0


class TestWCSStatus:
    """Tests for the WCSStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert WCSStatus.SUCCESS.value == "success"
        assert WCSStatus.FAILED.value == "failed"
        assert WCSStatus.INSUFFICIENT_SOURCES.value == "insufficient_sources"
        assert WCSStatus.TIMEOUT.value == "timeout"

    def test_status_is_string_enum(self):
        """Test that status values can be used as strings."""
        assert str(WCSStatus.SUCCESS) == "WCSStatus.SUCCESS"
        assert WCSStatus.SUCCESS == "success"


class TestWCSResult:
    """Tests for the WCSResult dataclass."""

    @pytest.fixture
    def sample_header(self):
        """Return a sample WCS header dictionary."""
        return {
            "CRVAL1": 180.0,
            "CRVAL2": 45.0,
            "CRPIX1": 512.0,
            "CRPIX2": 512.0,
            "CD1_1": -0.001,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 0.001,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
        }

    def test_from_fits_header(self, sample_header):
        """Test creating WCSResult from a FITS header."""
        wcs = WCSResult.from_fits_header(sample_header)
        assert wcs.crval1 == 180.0
        assert wcs.crval2 == 45.0
        assert wcs.crpix1 == 512.0
        assert wcs.crpix2 == 512.0
        assert wcs.cd_matrix == (-0.001, 0.0, 0.0, 0.001)

    def test_center_properties(self, sample_header):
        """Test center_ra and center_dec properties."""
        wcs = WCSResult.from_fits_header(sample_header)
        assert wcs.center_ra == 180.0
        assert wcs.center_dec == 45.0

    def test_pixel_scale(self, sample_header):
        """Test pixel scale calculation."""
        wcs = WCSResult.from_fits_header(sample_header)
        # 0.001 deg = 3.6 arcsec
        assert abs(wcs.pixel_scale - 3.6) < 0.1

    def test_to_fits_header(self, sample_header):
        """Test converting back to FITS header."""
        wcs = WCSResult.from_fits_header(sample_header)
        header = wcs.to_fits_header()
        assert header["CRVAL1"] == 180.0
        assert header["CRVAL2"] == 45.0


class TestMatchedStar:
    """Tests for the MatchedStar dataclass."""

    def test_create_minimal(self):
        """Test creating a matched star with minimal info."""
        star = MatchedStar(x=100.0, y=200.0, ra=180.0, dec=45.0)
        assert star.x == 100.0
        assert star.y == 200.0
        assert star.ra == 180.0
        assert star.dec == 45.0
        assert star.magnitude is None
        assert star.catalog is None
        assert star.catalog_id is None

    def test_create_full(self):
        """Test creating a matched star with all info."""
        star = MatchedStar(
            x=100.0,
            y=200.0,
            ra=180.0,
            dec=45.0,
            magnitude=12.5,
            catalog="Gaia",
            catalog_id="123456789",
        )
        assert star.magnitude == 12.5
        assert star.catalog == "Gaia"
        assert star.catalog_id == "123456789"


class TestSolveResult:
    """Tests for the SolveResult dataclass."""

    def test_failed_result(self):
        """Test creating a failed solve result."""
        result = SolveResult(
            success=False,
            status=WCSStatus.FAILED,
            wcs=None,
            matched_stars=[],
            detections=[Detection(x=100.0, y=200.0)],
            image_metadata=ImageMetadata(width=1024, height=1024),
        )
        assert result.success is False
        assert result.status == WCSStatus.FAILED
        assert result.wcs is None
        assert len(result.matched_stars) == 0

    def test_insufficient_sources_result(self):
        """Test creating an insufficient sources result."""
        result = SolveResult(
            success=False,
            status=WCSStatus.INSUFFICIENT_SOURCES,
            wcs=None,
            matched_stars=[],
            detections=[Detection(x=100.0, y=200.0)],
            image_metadata=ImageMetadata(width=1024, height=1024),
        )
        assert result.success is False
        assert result.status == WCSStatus.INSUFFICIENT_SOURCES
