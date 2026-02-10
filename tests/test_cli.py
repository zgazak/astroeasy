"""Tests for astroeasy CLI."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from astroeasy.cli import (
    cmd_build_config,
    cmd_examples,
    cmd_indices_examine,
    cmd_test_install,
    infer_index_series,
    main,
    _load_detections_from_file,
    _is_numeric,
)
from astroeasy.constants import AstrometryIndexSeries


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


class TestLoadDetectionsFromFile:
    """Tests for detection file loading."""

    def test_load_basic_csv(self):
        """Test loading a basic CSV file with header."""
        csv_content = """x,y,flux
100.0,200.0,1000.0
300.0,400.0,800.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            detections = _load_detections_from_file(str(csv_path))
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            detections = _load_detections_from_file(str(csv_path))
            assert len(detections) == 1
            assert detections[0].x == 100.0
        finally:
            csv_path.unlink()

    def test_load_csv_without_flux(self):
        """Test loading CSV without flux column."""
        csv_content = """x,y
100.0,200.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            detections = _load_detections_from_file(str(csv_path))
            assert len(detections) == 1
            assert detections[0].flux is None
        finally:
            csv_path.unlink()

    def test_load_tab_separated_no_header(self):
        """Test loading tab-separated file without header."""
        tsv_content = "100.0\t200.0\t1000.0\n300.0\t400.0\t800.0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(tsv_content)
            tsv_path = Path(f.name)

        try:
            detections = _load_detections_from_file(str(tsv_path))
            assert len(detections) == 2
            assert detections[0].x == 100.0
            assert detections[0].y == 200.0
            assert detections[0].flux == 1000.0
        finally:
            tsv_path.unlink()

    def test_load_tab_separated_scientific_notation(self):
        """Test loading tab-separated file with scientific notation."""
        tsv_content = (
            "1.254324757301874058e+02\t5.681827538918258824e+02\t9.179097860573370708e+05\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(tsv_content)
            tsv_path = Path(f.name)

        try:
            detections = _load_detections_from_file(str(tsv_path))
            assert len(detections) == 1
            assert abs(detections[0].x - 125.4324757301874058) < 1e-10
            assert abs(detections[0].y - 568.1827538918258824) < 1e-10
            assert abs(detections[0].flux - 917909.7860573370708) < 1e-3
        finally:
            tsv_path.unlink()

    def test_load_tab_separated_with_header(self):
        """Test loading tab-separated file with header."""
        tsv_content = "x\ty\tflux\n100.0\t200.0\t1000.0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(tsv_content)
            tsv_path = Path(f.name)

        try:
            detections = _load_detections_from_file(str(tsv_path))
            assert len(detections) == 1
            assert detections[0].x == 100.0
        finally:
            tsv_path.unlink()

    def test_load_csv_with_counts_column(self):
        """Test loading CSV with 'counts' column name."""
        csv_content = """x,y,counts
100.0,200.0,1000.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            detections = _load_detections_from_file(str(csv_path))
            assert len(detections) == 1
            assert detections[0].flux == 1000.0
        finally:
            csv_path.unlink()

    def test_load_real_test_file(self):
        """Test loading the real test data file."""
        test_file = Path(__file__).parent / "data" / "x_y_counts_1024_1024.txt"
        if test_file.exists():
            detections = _load_detections_from_file(str(test_file))
            assert len(detections) == 100
            # First row should be the brightest star
            assert abs(detections[0].x - 125.4324757301874058) < 1e-10


class TestIsNumeric:
    """Tests for the _is_numeric helper function."""

    def test_integer(self):
        assert _is_numeric("123") is True

    def test_float(self):
        assert _is_numeric("123.456") is True

    def test_scientific_notation(self):
        assert _is_numeric("1.254324757301874058e+02") is True

    def test_negative(self):
        assert _is_numeric("-123.456") is True

    def test_text(self):
        assert _is_numeric("x") is False

    def test_header(self):
        assert _is_numeric("flux") is False


class TestInferIndexSeries:
    """Tests for index series inference."""

    def test_nonexistent_path(self):
        """Test that nonexistent path returns CUSTOM."""
        result = infer_index_series(Path("/nonexistent/path"))
        assert result == AstrometryIndexSeries.SERIES_CUSTOM

    def test_empty_directory(self):
        """Test that empty directory returns CUSTOM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = infer_index_series(Path(tmpdir))
            assert result == AstrometryIndexSeries.SERIES_CUSTOM

    def test_infer_4100_series(self):
        """Test inferring 4100 series from file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create fake 4100 index files
            (tmppath / "index-4107.fits").touch()
            (tmppath / "index-4108.fits").touch()
            result = infer_index_series(tmppath)
            assert result == AstrometryIndexSeries.SERIES_4100

    def test_infer_4200_series(self):
        """Test inferring 4200 series from file names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create fake 4200 index files
            (tmppath / "index-4200-00.fits").touch()
            (tmppath / "index-4200-01.fits").touch()
            result = infer_index_series(tmppath)
            assert result == AstrometryIndexSeries.SERIES_4200

    @pytest.mark.skipif(
        not Path("/stars/data/share/5000/5200-LITE").exists(),
        reason="5200-LITE indices not available",
    )
    def test_infer_5200_lite(self):
        """Test inferring 5200_LITE series from real files."""
        result = infer_index_series(Path("/stars/data/share/5000/5200-LITE"))
        assert result == AstrometryIndexSeries.SERIES_5200_LITE


class TestCmdBuildConfig:
    """Tests for the build-config command."""

    def test_build_config_non_interactive(self):
        """Test building config in non-interactive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test-config.yaml"
            indices_dir = Path(tmpdir) / "indices"
            indices_dir.mkdir()

            class Args:
                indices_path = str(indices_dir)
                indices_series = "CUSTOM"
                docker_image = "test-image"
                scale_low = 0.5
                scale_high = 5.0
                cpulimit = 60
                max_sources = 200
                output = str(output_file)
                non_interactive = True

            result = cmd_build_config(Args())
            assert result == 0
            assert output_file.exists()

            # Verify file contents
            import yaml

            with open(output_file) as f:
                config = yaml.safe_load(f)
            assert config["docker_image"] == "test-image"
            assert config["cpulimit_seconds"] == 60
            assert config["max_sources"] == 200

    def test_build_config_requires_indices_path_non_interactive(self):
        """Test that non-interactive mode requires indices-path."""

        class Args:
            indices_path = None
            indices_series = None
            docker_image = None
            scale_low = 0.1
            scale_high = 10.0
            cpulimit = 30
            max_sources = 100
            output = None
            non_interactive = True

        result = cmd_build_config(Args())
        assert result == 1


class TestCmdExamples:
    """Tests for the examples command."""

    def test_examples_returns_zero(self):
        """Test that examples command returns exit code 0."""

        class Args:
            pass

        result = cmd_examples(Args())
        assert result == 0


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
