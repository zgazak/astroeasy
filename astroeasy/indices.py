"""Astrometry.net index file management."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from astropy.io import fits
from tqdm import tqdm

from astroeasy.constants import (
    ASTROMETRY_4100_EXPECTED_STRUCTURE,
    ASTROMETRY_4200_EXPECTED_STRUCTURE,
    ASTROMETRY_5200_EXPECTED_STRUCTURE,
    ASTROMETRY_5200_LITE_EXPECTED_STRUCTURE,
    ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE,
    ASTROMETRY_INDICES_URL_4100,
    ASTROMETRY_INDICES_URL_4200,
    ASTROMETRY_INDICES_URL_5200,
    ASTROMETRY_INDICES_URL_5200_LITE,
    AstrometryIndexSeries,
)

logger = logging.getLogger(__name__)


def human_readable_size(size_bytes: int) -> str:
    """Convert size in bytes to human-readable format with appropriate units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def get_fits_files(base_url: str) -> list[str]:
    """Get list of FITS files from a URL.

    Args:
        base_url: URL to scan for FITS files.

    Returns:
        List of full URLs to FITS files.
    """
    with urlopen(base_url) as response:
        html = response.read().decode("utf-8")

    # Exclude files matching index-##m#-*.fits pattern (multi-scale)
    fits_files = [
        base_url + filename
        for filename in re.findall(r'href="([^"]+\.fits)"', html)
        if not re.match(r"index-\d+m\d+-.*\.fits", filename)
    ]

    return fits_files


def download_fits_files(
    base_url: str,
    output_dir: str | Path | None = None,
    max_workers: int = 5,
) -> None:
    """Download .fits files, skipping existing files of the same size.

    Args:
        base_url: The URL to download .fits files from.
        output_dir: Directory to save files to. Defaults to current directory.
        max_workers: Number of concurrent downloads. Defaults to 5.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fits_files = get_fits_files(base_url)

    if not fits_files:
        logger.warning("No .fits files found!")
        return

    logger.info(f"Found {len(fits_files)} .fits files")

    def download_file(url):
        try:
            filename = url.split("/")[-1]
            if output_dir:
                filename = os.path.join(output_dir, filename)

            # Check if file exists
            if os.path.exists(filename):
                # Get remote file size
                req = Request(url, method="HEAD")
                with urlopen(req) as response:
                    remote_size = int(response.headers["Content-Length"])

                # Get local file size
                local_size = os.path.getsize(filename)

                if remote_size == local_size:
                    return
                else:
                    tqdm.write(f"Size mismatch for {filename}, downloading again...")
                    tqdm.write(f"Remote: {remote_size} bytes, Local: {local_size} bytes")
            else:
                tqdm.write(f"Downloading new file {filename}...")

            # Download with progress bar
            req = Request(url, method="HEAD")
            with urlopen(req) as response:
                file_size = int(response.headers["Content-Length"])

            with urlopen(url) as response:
                with open(filename, "wb") as f:
                    with tqdm(total=file_size, unit="B", unit_scale=True, desc=filename, leave=False) as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))

            tqdm.write(f"Successfully downloaded {filename}")
        except Exception as e:
            tqdm.write(f"Error downloading {url}: {e}")

    # Overall progress bar for all files
    with tqdm(total=len(fits_files), desc="Total progress", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for url in fits_files:
                future = executor.submit(download_file, url)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)


def extract_expected_structure(base_url: str) -> dict[str, int] | None:
    """Extract expected file sizes from remote server.

    Args:
        base_url: URL to scan for FITS files.

    Returns:
        Dictionary mapping filename to expected size in bytes.
    """
    fits_files = get_fits_files(base_url)

    if not fits_files:
        logger.warning("No .fits files found!")
        return None

    logger.info(f"Found {len(fits_files)} .fits files")

    results_dict = {}

    for url in tqdm(fits_files, desc="Extracting expected filesizes", ascii=True):
        filename = url.split("/")[-1]

        req = Request(url, method="HEAD")
        with urlopen(req) as response:
            remote_size = int(response.headers["Content-Length"])
            results_dict[filename] = remote_size

    return results_dict


def examine_by_path_and_structure(
    series: AstrometryIndexSeries | str,
    indices_path: str | Path,
    expected_structure: dict[str, int],
) -> bool:
    """Validate index files against expected structure.

    Args:
        series: Index series name for logging.
        indices_path: Path to the indices directory.
        expected_structure: Expected filename -> size mapping.

    Returns:
        True if all files are present and valid, False otherwise.
    """
    missing_indices = []
    size_mismatch_indices = []

    if series == AstrometryIndexSeries.SERIES_CUSTOM:
        logger.warning(
            f"[{series}] Astrometry indices are custom, skipping validation. "
            "Please consider adding this to the codebase."
        )
        return True

    for filename, expected_size in expected_structure.items():
        filepath = Path(indices_path) / filename
        if filepath.exists():
            local_size = os.path.getsize(filepath)

            if expected_size != local_size:
                size_mismatch_indices.append(filename)
        else:
            missing_indices.append(filename)

    complete_set = len(size_mismatch_indices) + len(missing_indices) == 0

    if complete_set:
        logger.info(f"[{series}] Astrometry indices [{series}] are complete and valid [{indices_path}]")
        return True

    if len(size_mismatch_indices) > 0:
        logger.warning(f"[{series}] Astrometry indices size mismatch for {', '.join(size_mismatch_indices)}")
    if len(missing_indices) > 0:
        logger.warning(f"[{series}] Astrometry indices missing: {', '.join(missing_indices)}")

    logger.warning(f"[{series}] Astrometry indices are incomplete")
    logger.warning(
        f"[{series}] fix: astroeasy indices download --series {series} --output {indices_path}"
    )

    return False


def pare_5200_to_SENPAI(
    series: AstrometryIndexSeries | str,
    indices_path: str | Path,
    output_path: str | Path,
) -> None:
    """Create 5200-SENPAI series from full 5200 series.

    This creates a reduced version of the 5200 series with only essential
    columns for matching (ra, dec, mag, ref_cat, ref_id).

    Args:
        series: Must be SERIES_5200.
        indices_path: Path to the full 5200 indices.
        output_path: Path to write the reduced indices.
    """
    if series != AstrometryIndexSeries.SERIES_5200:
        raise ValueError("Only series=5200 is supported for creating 5200-SENPAI")

    # Check if output already exists
    already_converted = examine_by_path_and_structure(
        "5200_SENPAI", output_path, ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE
    )
    if already_converted:
        logger.info("5200_SENPAI series already exists.")
        return

    source_indices_good = examine_by_path_and_structure(series, indices_path, ASTROMETRY_5200_EXPECTED_STRUCTURE)

    if not source_indices_good:
        raise ValueError("Source indices are not valid")

    columns_to_keep = ["ra", "dec", "mag", "ref_cat", "ref_id"]
    catalog_hdu_num = 13

    indices_path = Path(indices_path)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    total_original_size = 0
    total_new_size = 0

    for index_file in ASTROMETRY_5200_EXPECTED_STRUCTURE.keys():
        index_file_path = indices_path / index_file
        output_file_path = output_path / index_file

        if output_file_path.exists():
            if os.path.getsize(output_file_path) == ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE[index_file]:
                logger.info(f"[{index_file}] already exists and is valid")
                continue

        with fits.open(index_file_path) as hdul:
            catalog_hdu = hdul[catalog_hdu_num]

            # Create a new table with only the columns we want to keep
            new_data = fits.BinTableHDU.from_columns([catalog_hdu.data.columns[col] for col in columns_to_keep])
            new_data.header["AN_FILE"] = "TAGALONG"

            hdul[catalog_hdu_num] = new_data
            hdul.writeto(output_file_path, overwrite=True)

            original_size = ASTROMETRY_5200_EXPECTED_STRUCTURE[index_file]
            new_size = os.path.getsize(output_file_path)

            total_original_size += original_size
            total_new_size += new_size

            reduction = original_size - new_size
            reduction_percent = (reduction / original_size) * 100

            logger.info(
                f"{index_file} reduced by {human_readable_size(reduction)} ({reduction_percent:.1f}%) "
                f"from {human_readable_size(original_size)} to {human_readable_size(new_size)}"
            )

    if total_original_size > 0:
        total_reduction = total_original_size - total_new_size
        total_reduction_percent = (total_reduction / total_original_size) * 100

        logger.info(
            f"Total size reduced by {human_readable_size(total_reduction)} ({total_reduction_percent:.1f}%) "
            f"from {human_readable_size(total_original_size)} to {human_readable_size(total_new_size)}"
        )

    examine_by_path_and_structure("5200-SENPAI", output_path, ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE)


def get_expected_structure(
    series: AstrometryIndexSeries,
) -> tuple[list[str], dict[str, int]]:
    """Get download URLs and expected file structure for an index series.

    Args:
        series: The index series to get structure for.

    Returns:
        Tuple of (list of base URLs, expected filename -> size mapping).
    """
    if series == AstrometryIndexSeries.SERIES_5200_SENPAI:
        base_urls = []
        expected_structure = ASTROMETRY_5200_SENPAI_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_5200:
        base_urls = [ASTROMETRY_INDICES_URL_5200]
        expected_structure = ASTROMETRY_5200_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_5200_LITE:
        base_urls = [ASTROMETRY_INDICES_URL_5200_LITE]
        expected_structure = ASTROMETRY_5200_LITE_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_4100:
        base_urls = [ASTROMETRY_INDICES_URL_4100]
        expected_structure = ASTROMETRY_4100_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_4200:
        base_urls = [ASTROMETRY_INDICES_URL_4200]
        expected_structure = ASTROMETRY_4200_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_5200_LITE_4100:
        base_urls = [ASTROMETRY_INDICES_URL_5200_LITE, ASTROMETRY_INDICES_URL_4100]
        expected_structure = ASTROMETRY_5200_LITE_EXPECTED_STRUCTURE | ASTROMETRY_4100_EXPECTED_STRUCTURE
    elif series == AstrometryIndexSeries.SERIES_CUSTOM:
        base_urls = []
        expected_structure = {}
    else:
        raise AttributeError(f"Unknown series {series}")

    return base_urls, expected_structure


def examine_indices(
    indices_path: str | Path,
    series: AstrometryIndexSeries = AstrometryIndexSeries.SERIES_5200_LITE,
) -> bool:
    """Examine indices at the given path for completeness.

    Args:
        indices_path: Path to the indices directory.
        series: Which index series to validate against.

    Returns:
        True if indices are complete and valid.
    """
    _, expected_structure = get_expected_structure(series)
    return examine_by_path_and_structure(series, indices_path, expected_structure)


def download_indices(
    output_path: str | Path,
    series: AstrometryIndexSeries = AstrometryIndexSeries.SERIES_5200_LITE,
    max_workers: int = 5,
) -> None:
    """Download index files for the specified series.

    Args:
        output_path: Directory to download indices to.
        series: Which index series to download.
        max_workers: Number of concurrent downloads.
    """
    base_urls, _ = get_expected_structure(series)

    if not base_urls:
        raise ValueError(f"No download URLs available for series {series}")

    for base_url in base_urls:
        download_fits_files(base_url, output_dir=output_path, max_workers=max_workers)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Astrometry indices management")
    parser.add_argument(
        "action",
        choices=["download", "examine", "map_expected", "supported", "build_5200_senpai"],
        help="Action to perform",
    )
    parser.add_argument(
        "--series",
        type=AstrometryIndexSeries,
        choices=list(AstrometryIndexSeries),
        required=False,
        default=AstrometryIndexSeries.SERIES_5200_LITE,
        help="Index series to use",
    )
    parser.add_argument("--index_path", required=False, help="Path to the indices directory")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent downloads")

    args = parser.parse_args()

    base_urls, expected_structure = get_expected_structure(args.series)

    if args.action == "download":
        for base_url in base_urls:
            download_fits_files(base_url, output_dir=args.index_path, max_workers=args.workers)

    elif args.action == "examine":
        examine_by_path_and_structure(args.series, args.index_path, expected_structure)

    elif args.action == "supported":
        print(f"Supported indices: {', '.join(str(s) for s in AstrometryIndexSeries)}")

    elif args.action == "map_expected":
        expected = {}

        for series, url in zip(
            ["4100", "4200", "5200", "5200_LITE"],
            [
                ASTROMETRY_INDICES_URL_4100,
                ASTROMETRY_INDICES_URL_4200,
                ASTROMETRY_INDICES_URL_5200,
                ASTROMETRY_INDICES_URL_5200_LITE,
            ],
            strict=False,
        ):
            if series == str(args.series):
                expected[series] = extract_expected_structure(url)

        print(json.dumps(expected, indent=4))

    elif args.action == "build_5200_senpai":
        indices_path = Path(args.index_path)
        output_path = indices_path.parent / "5200-SENPAI"
        pare_5200_to_SENPAI(args.series, indices_path, output_path)
