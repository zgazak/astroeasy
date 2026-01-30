"""Command-line interface for astroeasy."""

import argparse
import csv
import logging
import sys
from pathlib import Path

from astroeasy.config import AstrometryConfig
from astroeasy.constants import AstrometryIndexSeries
from astroeasy.indices import download_indices, examine_indices
from astroeasy.models import Detection, ImageMetadata
from astroeasy.runner import solve_field, test_install


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def cmd_indices_download(args: argparse.Namespace) -> int:
    """Handle 'indices download' command."""
    series = AstrometryIndexSeries(args.series)
    output_path = Path(args.output)

    print(f"Downloading {series} indices to {output_path}")
    download_indices(output_path, series, max_workers=args.workers)
    print("Download complete.")
    return 0


def cmd_indices_examine(args: argparse.Namespace) -> int:
    """Handle 'indices examine' command."""
    series = AstrometryIndexSeries(args.series)
    path = Path(args.path)

    if examine_indices(path, series):
        print(f"Indices at {path} are complete and valid for {series}.")
        return 0
    else:
        print(f"Indices at {path} are incomplete for {series}.")
        return 1


def cmd_solve(args: argparse.Namespace) -> int:
    """Handle 'solve' command."""
    # Load config
    if args.config:
        config = AstrometryConfig.from_yaml(args.config)
    else:
        if not args.indices_path:
            print("Error: --indices-path is required when not using --config")
            return 1
        config = AstrometryConfig(
            indices_path=Path(args.indices_path),
            docker_image=args.docker_image,
            min_width_degrees=args.scale_low,
            max_width_degrees=args.scale_high,
            cpulimit_seconds=args.cpulimit,
        )

    # Load detections
    if args.xylist:
        detections = _load_detections_from_csv(args.xylist)
        if not detections:
            print(f"Error: No detections loaded from {args.xylist}")
            return 1
        if not args.width or not args.height:
            print("Error: --width and --height are required with --xylist")
            return 1
        metadata = ImageMetadata(
            width=args.width,
            height=args.height,
            boresight_ra=args.ra,
            boresight_dec=args.dec,
        )
    elif args.image:
        # Load image and extract detections (requires astropy)
        detections, metadata = _load_detections_from_fits(args.image)
        if not detections:
            print(f"Error: No detections extracted from {args.image}")
            return 1
    else:
        print("Error: Either --image or --xylist is required")
        return 1

    print(f"Solving {len(detections)} sources...")
    result = solve_field(detections, metadata, config)

    if result.success:
        print(f"Solved!")
        print(f"  Center RA:  {result.wcs.center_ra:.6f} deg")
        print(f"  Center Dec: {result.wcs.center_dec:.6f} deg")
        print(f"  Pixel scale: {result.wcs.pixel_scale:.3f} arcsec/pix")
        print(f"  Matched stars: {len(result.matched_stars)}")
        return 0
    else:
        print(f"Failed to solve: {result.status.value}")
        return 1


def cmd_test_install(args: argparse.Namespace) -> int:
    """Handle 'test-install' command."""
    docker_image = args.docker if args.docker else None

    if docker_image:
        print(f"Testing Docker installation: {docker_image}")
    else:
        print("Testing local installation")

    if test_install(docker_image):
        print("Installation test passed!")
        return 0
    else:
        print("Installation test failed.")
        return 1


def _load_detections_from_csv(path: str) -> list[Detection]:
    """Load detections from a CSV file.

    Expected columns: x, y, flux (optional)
    """
    detections = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row.get("x", row.get("X", 0)))
            y = float(row.get("y", row.get("Y", 0)))
            flux = None
            if "flux" in row:
                flux = float(row["flux"])
            elif "FLUX" in row:
                flux = float(row["FLUX"])
            detections.append(Detection(x=x, y=y, flux=flux))
    return detections


def _load_detections_from_fits(path: str) -> tuple[list[Detection], ImageMetadata]:
    """Load detections from a FITS image using simple source extraction.

    This uses a basic threshold-based detection. For production use,
    consider using a proper source extractor.
    """
    try:
        from astropy.io import fits
        import numpy as np
    except ImportError:
        print("Error: astropy is required for FITS image processing")
        return [], ImageMetadata(width=0, height=0)

    with fits.open(path) as hdul:
        # Find the image HDU
        image_data = None
        for hdu in hdul:
            if hdu.data is not None and len(hdu.data.shape) == 2:
                image_data = hdu.data.astype(float)
                break

        if image_data is None:
            print("Error: No 2D image data found in FITS file")
            return [], ImageMetadata(width=0, height=0)

        height, width = image_data.shape
        metadata = ImageMetadata(width=width, height=height)

        # Simple threshold detection
        # Calculate background statistics
        median = np.nanmedian(image_data)
        std = np.nanstd(image_data)
        threshold = median + 5 * std

        # Find pixels above threshold
        y_coords, x_coords = np.where(image_data > threshold)

        # Simple peak finding - group nearby pixels
        detections = []
        if len(x_coords) > 0:
            # For simplicity, just take bright pixels as detections
            # A real implementation would use proper source extraction
            fluxes = image_data[y_coords, x_coords]
            for x, y, flux in zip(x_coords, y_coords, fluxes):
                detections.append(Detection(x=float(x), y=float(y), flux=float(flux)))

            # Sort by flux and limit
            detections.sort(key=lambda d: d.flux or 0, reverse=True)
            detections = detections[:500]  # Limit to brightest sources

        return detections, metadata


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="astroeasy",
        description="Astrometry.net made easy - plate solving and index management",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # indices subcommand
    indices_parser = subparsers.add_parser("indices", help="Manage astrometry.net index files")
    indices_subparsers = indices_parser.add_subparsers(dest="indices_command")

    # indices download
    download_parser = indices_subparsers.add_parser("download", help="Download index files")
    download_parser.add_argument(
        "--series",
        type=str,
        default="5200_LITE",
        choices=[s.value for s in AstrometryIndexSeries],
        help="Index series to download (default: 5200_LITE)",
    )
    download_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for indices",
    )
    download_parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent downloads (default: 5)",
    )

    # indices examine
    examine_parser = indices_subparsers.add_parser("examine", help="Examine index files")
    examine_parser.add_argument(
        "--series",
        type=str,
        default="5200_LITE",
        choices=[s.value for s in AstrometryIndexSeries],
        help="Index series to validate (default: 5200_LITE)",
    )
    examine_parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to indices directory",
    )

    # solve subcommand
    solve_parser = subparsers.add_parser("solve", help="Solve astrometry for an image or source list")
    solve_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file",
    )
    solve_parser.add_argument(
        "--image",
        type=str,
        help="FITS image to solve",
    )
    solve_parser.add_argument(
        "--xylist",
        type=str,
        help="CSV file with x,y,flux columns",
    )
    solve_parser.add_argument("--width", type=int, help="Image width in pixels")
    solve_parser.add_argument("--height", type=int, help="Image height in pixels")
    solve_parser.add_argument("--ra", type=float, help="Approximate RA hint (degrees)")
    solve_parser.add_argument("--dec", type=float, help="Approximate Dec hint (degrees)")
    solve_parser.add_argument(
        "--indices-path",
        type=str,
        help="Path to astrometry.net index files",
    )
    solve_parser.add_argument(
        "--docker-image",
        type=str,
        help="Docker image for astrometry.net (omit for local)",
    )
    solve_parser.add_argument(
        "--scale-low",
        type=float,
        default=0.1,
        help="Minimum field width in degrees (default: 0.1)",
    )
    solve_parser.add_argument(
        "--scale-high",
        type=float,
        default=10.0,
        help="Maximum field width in degrees (default: 10.0)",
    )
    solve_parser.add_argument(
        "--cpulimit",
        type=int,
        default=30,
        help="CPU time limit in seconds (default: 30)",
    )

    # test-install subcommand
    test_parser = subparsers.add_parser("test-install", help="Test astrometry.net installation")
    test_parser.add_argument(
        "--docker",
        type=str,
        help="Docker image to test (omit to test local installation)",
    )
    test_parser.add_argument(
        "--local",
        action="store_true",
        help="Test local installation (default if --docker not specified)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "indices":
        if args.indices_command == "download":
            return cmd_indices_download(args)
        elif args.indices_command == "examine":
            return cmd_indices_examine(args)
        else:
            indices_parser.print_help()
            return 1
    elif args.command == "solve":
        return cmd_solve(args)
    elif args.command == "test-install":
        return cmd_test_install(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
