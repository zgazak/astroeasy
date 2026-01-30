import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_against_local(
    solve_field_command: str,
    run_directory: Path,
    verify_command: str | None = None,
):
    # run the container, mounting in config.indices_path to /usr/local/astrometry/
    # and building and mounting in a temporary directory (with config, etc)

    # Run the first solve-field command
    _ = subprocess.run(
        solve_field_command.split(),
        capture_output=True,
        text=True,
        cwd=run_directory,
    )

    successful_fit = (run_directory / "sources.match").exists()

    logger.info(f"[run_against_local] successful_fit: {successful_fit}")
    if verify_command and successful_fit:
        # Run the verification command
        _ = subprocess.run(
            verify_command.split(),
            capture_output=True,
            text=True,
            cwd=run_directory,
        )
        logger.info("[run_against_local] verification step complete")

    return successful_fit


def test_dotnet_install():
    try:
        # Check if solve-field exists in PATH using which/where command
        which_cmd = "where" if subprocess.os.name == "nt" else "which"
        location_result = subprocess.run([which_cmd, "solve-field"], capture_output=True, text=True)

        if location_result.returncode != 0:
            logger.error("solve-field (Astrometry.net) is not installed or not in PATH")
            return False

        solve_field_path = location_result.stdout.strip()
        logger.info(f"solve-field found at: {solve_field_path}")

        # Get version information
        result = subprocess.run(["solve-field"], capture_output=True, text=True)
        revision_match = re.search(r"Revision\s+([\d.]+)", result.stdout)

        if revision_match:
            version = revision_match.group(1)
            logger.info(f"solve-field (Astrometry.net) version: {version}")
            return True
        else:
            logger.warning("solve-field (Astrometry.net) is installed but version could not be determined")
            return True

    except FileNotFoundError:
        logger.error("solve-field (Astrometry.net) is not installed or not in PATH")
        return False


if __name__ == "__main__":
    test_dotnet_install()
