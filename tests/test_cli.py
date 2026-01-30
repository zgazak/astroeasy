"""Tests for astroeasy CLI."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from astroeasy.cli import (
    cmd_indices_examine,
    cmd_test_install,
    main,
    _load_detections_from_csv,
)


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_help_returns_zero(self):
        """Test that --help returns exit code 0."""
        with patch("sys.argv", ["astroeasy", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_no_args_returns_zero(self):
        """Test that no arguments returns exit code 0 (shows help)."""
        with patch("sys.argv", ["astroeasy"]):
            result = main()
            assert result == 0


class TestLoadDetectionsFromCSV:
    """Tests for CSV detection loading."""

    def test_load_basic_csv(self):
        """Test loading a basic CSV file."""
        csv_content = """x,y,flux
100.0,200.0,1000.0
300.0,400.0,800.0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            detections = _load_detections_from_csv(str(csv_path))
            assert len(detections) == 2
            assert detections[0].x == 100.0
            assert detections[0].y == 200.0
            assert detections[0].flux == 1000.0
        finally:
            csv_path.unlink()

    def test_load_csv_uppercase_columns(self):
        """Test loading CSV with uppercase column names."""
        csv_content = """X,Y,FLUX
100.0,200.0,1000.0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            detections = _load_detections_from_csv(str(csv_path))
            assert len(detections) == 1
            assert detections[0].x == 100.0
        finally:
            csv_path.unlink()

    def test_load_csv_without_flux(self):
        """Test loading CSV without flux column."""
        csv_content = """x,y
100.0,200.0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            detections = _load_detections_from_csv(str(csv_path))
            assert len(detections) == 1
            assert detections[0].flux is None
        finally:
            csv_path.unlink()


class TestCmdIndicesExamine:
    """Tests for the indices examine command."""

    def test_examine_nonexistent_path(self):
        """Test examining a nonexistent path returns 1."""

        class Args:
            series = "5200_LITE"
            path = "/nonexistent/path"

        result = cmd_indices_examine(Args())
        assert result == 1

    @pytest.mark.skipif(
        not Path("/stars/data/share/5000/5200-LITE").exists(),
        reason="5200-LITE indices not available",
    )
    def test_examine_valid_path(self):
        """Test examining valid indices returns 0."""

        class Args:
            series = "5200_LITE"
            path = "/stars/data/share/5000/5200-LITE"

        result = cmd_indices_examine(Args())
        assert result == 0


class TestCmdTestInstall:
    """Tests for the test-install command."""

    def test_test_install_local(self):
        """Test the test-install command for local installation."""

        class Args:
            docker = None

        # This will pass or fail depending on local installation
        result = cmd_test_install(Args())
        assert result in (0, 1)

    def test_test_install_docker_invalid(self):
        """Test the test-install command with invalid docker image."""

        class Args:
            docker = "nonexistent-image-12345"

        result = cmd_test_install(Args())
        assert result == 1
