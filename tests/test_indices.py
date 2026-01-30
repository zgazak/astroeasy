"""Tests for astroeasy index management."""

import tempfile
from pathlib import Path

import pytest

from astroeasy.constants import (
    ASTROMETRY_5200_LITE_EXPECTED_STRUCTURE,
    AstrometryIndexSeries,
)
from astroeasy.indices import (
    examine_indices,
    get_expected_structure,
    human_readable_size,
)


class TestHumanReadableSize:
    """Tests for the human_readable_size function."""

    def test_bytes(self):
        """Test formatting bytes."""
        assert human_readable_size(500) == "500.00 B"

    def test_kilobytes(self):
        """Test formatting kilobytes."""
        assert human_readable_size(1024) == "1.00 KB"
        assert human_readable_size(1536) == "1.50 KB"

    def test_megabytes(self):
        """Test formatting megabytes."""
        assert human_readable_size(1024 * 1024) == "1.00 MB"

    def test_gigabytes(self):
        """Test formatting gigabytes."""
        assert human_readable_size(1024 * 1024 * 1024) == "1.00 GB"

    def test_large_size(self):
        """Test formatting large sizes."""
        # 15 GB
        size = 15 * 1024 * 1024 * 1024
        result = human_readable_size(size)
        assert "GB" in result
        assert "15" in result


class TestGetExpectedStructure:
    """Tests for the get_expected_structure function."""

    def test_5200_lite(self):
        """Test getting structure for 5200_LITE series."""
        urls, structure = get_expected_structure(AstrometryIndexSeries.SERIES_5200_LITE)
        assert len(urls) == 1
        assert "5200" in urls[0]
        assert "LITE" in urls[0]
        assert structure == ASTROMETRY_5200_LITE_EXPECTED_STRUCTURE

    def test_5200(self):
        """Test getting structure for 5200 series."""
        urls, structure = get_expected_structure(AstrometryIndexSeries.SERIES_5200)
        assert len(urls) == 1
        assert len(structure) > 0

    def test_4100(self):
        """Test getting structure for 4100 series."""
        urls, structure = get_expected_structure(AstrometryIndexSeries.SERIES_4100)
        assert len(urls) == 1
        assert len(structure) > 0
        # 4100 series files start with index-41
        assert any("4107" in f or "4108" in f for f in structure.keys())

    def test_combined_series(self):
        """Test getting structure for combined series."""
        urls, structure = get_expected_structure(
            AstrometryIndexSeries.SERIES_5200_LITE_4100
        )
        assert len(urls) == 2
        # Should have files from both series
        has_5200 = any("5200" in f for f in structure.keys())
        has_4100 = any("4107" in f or "4108" in f for f in structure.keys())
        assert has_5200 and has_4100

    def test_custom_series(self):
        """Test getting structure for custom series."""
        urls, structure = get_expected_structure(AstrometryIndexSeries.SERIES_CUSTOM)
        assert urls == []
        assert structure == {}

    def test_unknown_series(self):
        """Test that unknown series raises error."""
        with pytest.raises(AttributeError):
            get_expected_structure("unknown_series")


class TestExamineIndices:
    """Tests for the examine_indices function."""

    def test_nonexistent_path(self):
        """Test examining a nonexistent path."""
        result = examine_indices(
            Path("/nonexistent/path"),
            AstrometryIndexSeries.SERIES_5200_LITE,
        )
        assert result is False

    def test_empty_directory(self):
        """Test examining an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = examine_indices(
                Path(tmpdir),
                AstrometryIndexSeries.SERIES_5200_LITE,
            )
            assert result is False

    def test_custom_series_always_valid(self):
        """Test that custom series always returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = examine_indices(
                Path(tmpdir),
                AstrometryIndexSeries.SERIES_CUSTOM,
            )
            assert result is True

    @pytest.mark.skipif(
        not Path("/stars/data/share/5000/5200-LITE").exists(),
        reason="5200-LITE indices not available",
    )
    def test_valid_indices(self):
        """Test examining valid indices (requires indices to be present)."""
        result = examine_indices(
            Path("/stars/data/share/5000/5200-LITE"),
            AstrometryIndexSeries.SERIES_5200_LITE,
        )
        assert result is True
