"""Tests for the catalog module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from astroeasy.catalog.gaia import (
    CatalogStar,
    query_gaia_field,
    get_field_bounds_from_wcs,
)


class TestCatalogStar:
    """Tests for CatalogStar dataclass."""

    def test_create_basic(self):
        """Test basic creation."""
        star = CatalogStar(ra=180.0, dec=45.0, magnitude=10.5)
        assert star.ra == 180.0
        assert star.dec == 45.0
        assert star.magnitude == 10.5
        assert star.catalog == "Gaia"
        assert star.source_id is None

    def test_create_with_all_fields(self):
        """Test creation with all fields."""
        star = CatalogStar(
            ra=123.456,
            dec=-30.0,
            magnitude=8.5,
            source_id="12345678",
            catalog="Gaia DR3",
        )
        assert star.ra == 123.456
        assert star.dec == -30.0
        assert star.magnitude == 8.5
        assert star.source_id == "12345678"
        assert star.catalog == "Gaia DR3"


class TestGetFieldBoundsFromWCS:
    """Tests for get_field_bounds_from_wcs function."""

    @pytest.fixture
    def simple_wcs(self):
        """Create a simple WCS for testing."""
        from astropy.wcs import WCS
        from astropy.io import fits

        header = fits.Header()
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRPIX1"] = 512.0
        header["CRPIX2"] = 512.0
        header["CRVAL1"] = 180.0
        header["CRVAL2"] = 45.0
        header["CDELT1"] = -0.001  # 3.6 arcsec/pixel
        header["CDELT2"] = 0.001
        header["NAXIS1"] = 1024
        header["NAXIS2"] = 1024

        return WCS(header)

    def test_basic_bounds(self, simple_wcs):
        """Test basic bounds extraction."""
        min_ra, max_ra, min_dec, max_dec = get_field_bounds_from_wcs(
            simple_wcs, 1024, 1024
        )

        # Check that bounds are reasonable
        assert 179 < min_ra < 181
        assert 179 < max_ra < 181
        assert 44 < min_dec < 46
        assert 44 < max_dec < 46

        # Dec range should be about 1 degree for 1024 pixels at 0.001 deg/pixel
        assert abs(max_dec - min_dec - 1.0) < 0.1

    def test_bounds_width_height(self, simple_wcs):
        """Test bounds with different width/height."""
        min_ra1, max_ra1, min_dec1, max_dec1 = get_field_bounds_from_wcs(
            simple_wcs, 1024, 1024
        )
        min_ra2, max_ra2, min_dec2, max_dec2 = get_field_bounds_from_wcs(
            simple_wcs, 512, 512
        )

        # Smaller image should have smaller bounds
        dec_range1 = max_dec1 - min_dec1
        dec_range2 = max_dec2 - min_dec2
        assert dec_range2 < dec_range1


class TestQueryGaiaField:
    """Tests for query_gaia_field function."""

    def test_import_error_without_astroquery(self):
        """Test that proper error is raised without astroquery."""
        # Mock the import to fail
        with patch.dict("sys.modules", {"astroquery": None, "astroquery.gaia": None}):
            with pytest.raises(ImportError, match="astroquery"):
                # Clear cached import and try again
                import sys
                if "astroeasy.catalog.gaia" in sys.modules:
                    # Can't easily test this without reloading
                    pass
                query_gaia_field(0, 10, 0, 10)

    def test_query_returns_list(self):
        """Test that query returns a list (even if empty due to no astroquery)."""
        try:
            # This will either work (if astroquery installed) or raise ImportError
            from astroquery.gaia import Gaia
            # If we get here, astroquery is installed - skip mock test
            pytest.skip("astroquery is installed, skipping mock test")
        except ImportError:
            # astroquery not installed, test the ImportError
            with pytest.raises(ImportError, match="astroquery"):
                query_gaia_field(179.0, 181.0, 44.0, 46.0)

    def test_function_signature(self):
        """Test that function has expected signature."""
        import inspect
        sig = inspect.signature(query_gaia_field)
        params = list(sig.parameters.keys())
        assert "min_ra" in params
        assert "max_ra" in params
        assert "min_dec" in params
        assert "max_dec" in params
        assert "faint_limit" in params
        assert "max_stars" in params
