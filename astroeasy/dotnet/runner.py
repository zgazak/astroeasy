"""Core solve-field orchestration for astrometry.net."""

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from astroeasy.config import AstrometryConfig
from astroeasy.dotnet.docker import run_against_container
from astroeasy.dotnet.local import run_against_local
from astroeasy.models import (
    Detection,
    ImageMetadata,
    MatchedStar,
    SolveResult,
    WCSResult,
    WCSStatus,
)

logger = logging.getLogger(__name__)


def _write_astrometry_cfg(config: AstrometryConfig, indices_path: str, output_path: Path) -> None:
    """Write astrometry.net configuration file.

    Args:
        config: Astrometry configuration.
        indices_path: Path to index files (may differ from config for Docker).
        output_path: Path to write the config file.
    """
    cfg_content = f"""# Astrometry.net configuration
inparallel
add_path {indices_path}
autoindex

# Minimum and maximum quad size for solver
minwidth {config.min_width_degrees}
maxwidth {config.max_width_degrees}
"""
    with open(output_path, "w") as f:
        f.write(cfg_content)


def _write_sources_fits(
    detections: list[Detection],
    output_path: Path,
    max_sources: int = 100,
) -> None:
    """Write detections to a FITS file for astrometry.net.

    Args:
        detections: List of detections to write.
        output_path: Path to write the FITS file.
        max_sources: Maximum number of sources to include.
    """
    # Sort by flux (brightest first) and limit
    sorted_detections = sorted(
        [d for d in detections if d.flux is not None],
        key=lambda d: d.flux,
        reverse=True,
    )
    # Add detections without flux at the end
    sorted_detections.extend([d for d in detections if d.flux is None])
    limited_detections = sorted_detections[:max_sources]

    # Create FITS columns
    x_col = fits.Column(
        name="X",
        format="E",
        array=np.array([d.x for d in limited_detections], dtype=np.float32),
    )
    y_col = fits.Column(
        name="Y",
        format="E",
        array=np.array([d.y for d in limited_detections], dtype=np.float32),
    )
    flux_col = fits.Column(
        name="FLUX",
        format="E",
        array=np.array(
            [d.flux if d.flux is not None else 0.0 for d in limited_detections],
            dtype=np.float32,
        ),
    )

    coldefs = fits.ColDefs([x_col, y_col, flux_col])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.header["AN_FILE"] = "sources"

    primary = fits.PrimaryHDU()
    hdul = fits.HDUList([primary, hdu])
    hdul.writeto(output_path, overwrite=True)


def _read_wcs_result(wcs_path: Path) -> WCSResult:
    """Read WCS result from astrometry.net output.

    Args:
        wcs_path: Path to the .wcs FITS file.

    Returns:
        WCSResult instance.
    """
    with fits.open(wcs_path) as hdul:
        header = hdul[0].header
        return WCSResult.from_fits_header(header)


def _read_matched_stars(
    temp_path: Path,
    wcs_result: WCSResult,
) -> list[MatchedStar]:
    """Read matched stars from astrometry.net output.

    Args:
        temp_path: Directory containing astrometry.net output files.
        wcs_result: WCS solution for coordinate conversion.

    Returns:
        List of MatchedStar instances.
    """
    rdls_path = temp_path / "sources.rdls"
    if not rdls_path.exists():
        return []

    matched_stars = []
    world_coords = wcs_result.to_astropy_wcs()

    with fits.open(rdls_path) as hdul:
        ra_dec_data = hdul[1].data

        # Sort by magnitude if available
        if "mag" in ra_dec_data.names:
            ra_dec_data = np.sort(ra_dec_data, order="mag")

        for line in ra_dec_data:
            ra = float(line["RA"])
            dec = float(line["DEC"])

            # Handle optional fields
            magnitude = float(line["mag"]) if "mag" in ra_dec_data.names else None
            catalog = line["ref_cat"] if "ref_cat" in ra_dec_data.names else None
            catalog_id = str(line["ref_id"]) if "ref_id" in ra_dec_data.names else None

            # Convert RA/Dec to pixel coordinates using WCS
            x, y = world_coords.all_world2pix(ra, dec, 0)

            matched_stars.append(
                MatchedStar(
                    ra=ra,
                    dec=dec,
                    magnitude=magnitude,
                    catalog=catalog,
                    catalog_id=catalog_id,
                    x=float(x),
                    y=float(y),
                )
            )

    return matched_stars


def _write_existing_wcs(wcs: WCSResult, output_path: Path) -> None:
    """Write an existing WCS to a FITS file for verification.

    Args:
        wcs: Existing WCS solution.
        output_path: Path to write the WCS FITS file.
    """
    header = fits.Header()
    for key, value in wcs.raw_header.items():
        if key and not key.startswith("COMMENT") and not key.startswith("HISTORY"):
            try:
                header[key] = value
            except (ValueError, KeyError):
                pass

    primary = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary])
    hdul.writeto(output_path, overwrite=True)


def solve_field(
    detections: list[Detection],
    image_metadata: ImageMetadata,
    config: AstrometryConfig,
    existing_wcs: WCSResult | None = None,
) -> SolveResult:
    """Solve astrometry for a set of detections.

    Args:
        detections: List of detected sources with x, y coordinates.
        image_metadata: Metadata about the image (width, height, optional hints).
        config: Astrometry configuration.
        existing_wcs: Optional existing WCS to verify/refine.

    Returns:
        SolveResult with WCS solution and matched stars if successful.
    """
    logger.info(f"Attempting astrometric solution on {len(detections)} sources")

    # Determine runner and indices path
    if config.docker_image:
        runner = run_against_container
        indices_path = "/usr/local/astrometry/data"
    else:
        runner = run_against_local
        indices_path = str(config.indices_path)

    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = Path(temp_dir)

        # Write sources file
        sources_file = temp_path / "sources.fits"
        _write_sources_fits(detections, sources_file, config.max_sources)

        # Write astrometry config
        _write_astrometry_cfg(config, indices_path, temp_path / "astrometry.cfg")

        # Build width/height arguments
        width_height_args = f" --width {image_metadata.width} --height {image_metadata.height}"

        tweak_order = config.tweak_order
        if tweak_order > 0:
            tweak_command = f" --tweak-order {tweak_order}"
        else:
            tweak_command = " --no-tweak"

        if existing_wcs is not None:
            # Verify existing WCS
            wcs_file = temp_path / "sources.wcs"
            _write_existing_wcs(existing_wcs, wcs_file)
            solve_field_command = None
        else:
            # Build solve-field command
            solve_field_command = (
                f"timeout {config.cpulimit_seconds}s solve-field sources.fits --config astrometry.cfg --continue"
                " --x-column X --y-column Y"
            )
            solve_field_command += width_height_args

            # Add position hint if available
            if image_metadata.boresight_ra is not None and image_metadata.boresight_dec is not None:
                solve_field_command += (
                    f" --ra {image_metadata.boresight_ra}"
                    f" --dec {image_metadata.boresight_dec}"
                    " --radius 10.0"
                )

            solve_field_command += " --crpix-center --overwrite --scale-units degw"
            solve_field_command += f" --scale-low {config.min_width_degrees}"
            solve_field_command += f" --scale-high {config.max_width_degrees}"
            solve_field_command += f" --continue --cpulimit {config.cpulimit_seconds} --sort-column FLUX"
            solve_field_command += " --no-plots"
            solve_field_command += tweak_command

        # Build verification command
        verify_command = (
            f"solve-field --verify sources.wcs sources.fits{tweak_command} --tag-all --continue --no-plots"
            + width_height_args
        )

        # Run solver
        if config.docker_image:
            success = runner(
                solve_field_command,
                temp_path,
                config.indices_path,
                config.docker_image,
                verify_command,
                config.output_dir,
            )
        else:
            success = runner(solve_field_command, temp_path, verify_command)

        if success:
            wcs_result = _read_wcs_result(temp_path / "sources.wcs")
            matched_stars = _read_matched_stars(temp_path, wcs_result)
            status = WCSStatus.SUCCESS
        else:
            wcs_result = None
            matched_stars = []
            status = WCSStatus.FAILED

        return SolveResult(
            success=success,
            status=status,
            wcs=wcs_result,
            matched_stars=matched_stars,
            detections=detections,
            image_metadata=image_metadata,
        )

    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")
