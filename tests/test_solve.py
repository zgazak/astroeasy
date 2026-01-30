"""Integration tests for astroeasy plate solving."""

import pytest

from astroeasy import (
    AstrometryConfig,
    Detection,
    ImageMetadata,
    SolveResult,
    WCSStatus,
    solve_field,
)

from .conftest import requires_docker_install, requires_indices, requires_local_install


class TestSolveFieldBasic:
    """Basic tests for solve_field that don't require astrometry.net."""

    def test_insufficient_sources(self, minimal_detections, sample_metadata, local_config):
        """Test that solve_field returns INSUFFICIENT_SOURCES for too few detections."""
        result = solve_field(minimal_detections, sample_metadata, local_config)

        assert result.success is False
        assert result.status == WCSStatus.INSUFFICIENT_SOURCES
        assert result.wcs is None
        assert result.detections == minimal_detections
        assert result.image_metadata == sample_metadata

    def test_result_contains_input_data(
        self, sample_detections, sample_metadata, local_config
    ):
        """Test that result contains the input detections and metadata."""
        # Use a very short timeout so it fails quickly
        config = AstrometryConfig(
            indices_path=local_config.indices_path,
            cpulimit_seconds=1,
            docker_image=None,
        )

        result = solve_field(sample_detections, sample_metadata, config)

        # Regardless of success, result should contain input data
        assert result.detections == sample_detections
        assert result.image_metadata == sample_metadata


@requires_local_install
@requires_indices
class TestSolveFieldLocal:
    """Integration tests for solve_field with local astrometry.net."""

    def test_solve_success(
        self,
        sample_detections: list[Detection],
        sample_metadata: ImageMetadata,
        local_config: AstrometryConfig,
    ):
        """Test successful plate solving with local installation."""
        result = solve_field(sample_detections, sample_metadata, local_config)

        assert result.success is True
        assert result.status == WCSStatus.SUCCESS
        assert result.wcs is not None

        # Check WCS has reasonable values
        assert 0 <= result.wcs.center_ra <= 360
        assert -90 <= result.wcs.center_dec <= 90
        assert result.wcs.pixel_scale > 0

        # Should have matched some stars
        assert len(result.matched_stars) > 0

    def test_solve_with_hint(
        self,
        sample_detections: list[Detection],
        local_config: AstrometryConfig,
    ):
        """Test plate solving with position hint."""
        # First solve without hint to get the actual position
        metadata_no_hint = ImageMetadata(width=1024, height=1024)
        result1 = solve_field(sample_detections, metadata_no_hint, local_config)

        if not result1.success:
            pytest.skip("Initial solve failed, cannot test with hint")

        # Now solve with hint near the actual position
        metadata_with_hint = ImageMetadata(
            width=1024,
            height=1024,
            boresight_ra=result1.wcs.center_ra,
            boresight_dec=result1.wcs.center_dec,
        )
        result2 = solve_field(sample_detections, metadata_with_hint, local_config)

        assert result2.success is True
        # Results should be very similar
        assert abs(result1.wcs.center_ra - result2.wcs.center_ra) < 0.1
        assert abs(result1.wcs.center_dec - result2.wcs.center_dec) < 0.1


@requires_docker_install
@requires_indices
class TestSolveFieldDocker:
    """Integration tests for solve_field with Docker astrometry.net."""

    def test_solve_success(
        self,
        sample_detections: list[Detection],
        sample_metadata: ImageMetadata,
        docker_config: AstrometryConfig,
    ):
        """Test successful plate solving with Docker installation."""
        result = solve_field(sample_detections, sample_metadata, docker_config)

        assert result.success is True
        assert result.status == WCSStatus.SUCCESS
        assert result.wcs is not None

        # Check WCS has reasonable values
        assert 0 <= result.wcs.center_ra <= 360
        assert -90 <= result.wcs.center_dec <= 90
        assert result.wcs.pixel_scale > 0

        # Should have matched some stars
        assert len(result.matched_stars) > 0


class TestSolveResultWCS:
    """Tests for WCS result handling."""

    @requires_local_install
    @requires_indices
    def test_wcs_to_astropy(
        self,
        sample_detections: list[Detection],
        sample_metadata: ImageMetadata,
        local_config: AstrometryConfig,
    ):
        """Test converting WCSResult to astropy WCS."""
        result = solve_field(sample_detections, sample_metadata, local_config)

        if not result.success:
            pytest.skip("Solve failed")

        # Convert to astropy WCS
        astropy_wcs = result.wcs.to_astropy_wcs()

        # Should be able to do coordinate transforms
        ra, dec = astropy_wcs.all_pix2world(512, 512, 0)
        assert 0 <= ra <= 360
        assert -90 <= dec <= 90

    @requires_local_install
    @requires_indices
    def test_matched_stars_have_coordinates(
        self,
        sample_detections: list[Detection],
        sample_metadata: ImageMetadata,
        local_config: AstrometryConfig,
    ):
        """Test that matched stars have both pixel and sky coordinates."""
        result = solve_field(sample_detections, sample_metadata, local_config)

        if not result.success:
            pytest.skip("Solve failed")

        for star in result.matched_stars:
            # Should have pixel coordinates
            assert star.x >= 0
            assert star.y >= 0

            # Should have sky coordinates
            assert 0 <= star.ra <= 360
            assert -90 <= star.dec <= 90
