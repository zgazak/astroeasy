"""Image plotting with WCS grid overlays."""

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patheffects import withStroke

from astroeasy.plotting.normalization import zscale

logger = logging.getLogger(__name__)


def _font_size(img: np.ndarray) -> float:
    """Calculate appropriate font size based on image dimensions."""
    return max(6, min(img.shape[1], img.shape[0]) * 0.01)


def _format_ra(ra_deg: float) -> str:
    """Format RA in hours:minutes:seconds."""
    coord = SkyCoord(ra=ra_deg * u.deg, dec=0 * u.deg)
    text = coord.ra.to_string(unit=u.hour, sep=":", pad=True, precision=1)
    return text.replace(":", "h", 1).replace(":", "m", 1) + "s"


def _format_dec(dec_deg: float) -> str:
    """Format Dec in degrees:arcmin:arcsec."""
    coord = SkyCoord(ra=0 * u.deg, dec=dec_deg * u.deg)
    text = coord.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True)
    return text.replace(":", "\u00b0", 1).replace(":", "'", 1) + "''"


def plot_solved_field(
    image_path: str | Path,
    wcs_path: str | Path | None = None,
    output_path: str | Path | None = None,
    wcs: WCS | None = None,
    matched_stars: list | None = None,
    catalog_stars: list | None = None,
    show_grid: bool = True,
    show_matched: bool = True,
    show_catalog: bool = True,
    catalog_limit: int = 200,
    dpi: int | None = None,
    title: str | None = None,
) -> tuple | None:
    """Plot a solved field with WCS grid and optional star overlays.

    Args:
        image_path: Path to the FITS image file.
        wcs_path: Path to the WCS FITS file (if separate from image).
        output_path: Path to save the plot. If None, returns (fig, ax).
        wcs: Optional pre-loaded Astropy WCS object.
        matched_stars: List of MatchedStar objects from solve result.
        catalog_stars: List of CatalogStar objects from catalog query.
        show_grid: Whether to show RA/Dec grid lines (default True).
        show_matched: Whether to show matched stars (default True).
        show_catalog: Whether to show catalog stars (default True).
        catalog_limit: Maximum catalog stars to show (default 200).
        dpi: Output DPI (default auto-scales based on image size).
        title: Optional title for the plot.

    Returns:
        If output_path is None, returns (fig, ax). Otherwise returns None.
    """
    image_path = Path(image_path)

    # Load image data
    with fits.open(image_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and len(hdu.data.shape) == 2:
                img = hdu.data.astype(np.float32)
                break
        else:
            raise ValueError("No 2D image data found in FITS file")

    height, width = img.shape

    # Load WCS if not provided
    if wcs is None:
        if wcs_path is not None:
            wcs_path = Path(wcs_path)
            with fits.open(wcs_path) as hdul:
                wcs = WCS(hdul[0].header)
        else:
            # Try to get WCS from image header
            with fits.open(image_path) as hdul:
                for hdu in hdul:
                    try:
                        wcs = WCS(hdu.header)
                        if wcs.has_celestial:
                            break
                    except Exception:
                        continue
                else:
                    wcs = None

    # Determine DPI
    if dpi is None:
        max_dimension = max(height, width)
        dpi = 75 if max_dimension > 4000 else 150

    # Create figure
    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        frameon=False,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    ax = fig.add_subplot(111)
    ax.set_frame_on(False)

    # Display image with zscale
    ax.imshow(zscale(img), cmap="gray", origin="lower")

    # Draw WCS grid if available
    if wcs is not None and wcs.has_celestial and show_grid:
        _draw_wcs_grid(ax, wcs, img, width, height)

    # Plot matched stars
    if matched_stars and show_matched:
        _plot_matched_stars(ax, matched_stars, wcs)

    # Plot catalog stars
    if catalog_stars and show_catalog:
        _plot_catalog_stars(ax, catalog_stars[:catalog_limit], wcs)

    # Set axis limits
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title if specified
    if title:
        ax.set_title(title, color="white", fontsize=_font_size(img) * 1.5, pad=10)

    # Add astroeasy watermark
    shift = 0.005 * np.array(img.shape)
    try:
        from importlib.metadata import version
        ver = version("astroeasy")
    except Exception:
        ver = "dev"
    ax.text(
        width - shift[1],
        height - shift[0],
        f"astroeasy v{ver}",
        color="white",
        ha="right",
        va="top",
        size=_font_size(img) * 1.2,
        alpha=0.8,
    )

    if output_path:
        output_path = Path(output_path)
        save_kwargs = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0}

        ext = output_path.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            save_kwargs["format"] = "jpeg"
            save_kwargs["quality"] = 95
        elif ext == ".png":
            save_kwargs["format"] = "png"

        plt.savefig(output_path, **save_kwargs)
        plt.close()

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved plot to {output_path} ({file_size_mb:.1f} MB)")
        return None
    else:
        return fig, ax


def _calculate_line_angle(x_coords: np.ndarray, y_coords: np.ndarray, label_x: float, label_y: float) -> float:
    """Calculate the angle of a grid line at a given point.

    Args:
        x_coords: X coordinates of the line.
        y_coords: Y coordinates of the line.
        label_x: X position where we want the angle.
        label_y: Y position where we want the angle.

    Returns:
        Angle in degrees, normalized to [-90, 90].
    """
    if len(x_coords) < 2:
        return 0.0

    # Find the point closest to the label position
    dists = np.sqrt((x_coords - label_x) ** 2 + (y_coords - label_y) ** 2)
    closest_idx = np.argmin(dists)

    # Use neighbors to get direction
    if closest_idx > 0 and closest_idx < len(x_coords) - 1:
        dy = y_coords[closest_idx + 1] - y_coords[closest_idx - 1]
        dx = x_coords[closest_idx + 1] - x_coords[closest_idx - 1]
    elif closest_idx > 0:
        dy = y_coords[closest_idx] - y_coords[closest_idx - 1]
        dx = x_coords[closest_idx] - x_coords[closest_idx - 1]
    elif closest_idx < len(x_coords) - 1:
        dy = y_coords[closest_idx + 1] - y_coords[closest_idx]
        dx = x_coords[closest_idx + 1] - x_coords[closest_idx]
    else:
        dy = y_coords[-1] - y_coords[0]
        dx = x_coords[-1] - x_coords[0]

    # Calculate angle (negative because matplotlib y-axis may be inverted)
    angle = np.degrees(np.arctan2(dy, dx))

    # Normalize to [-90, 90] so text is always readable
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def _draw_wcs_grid(ax, wcs: WCS, img: np.ndarray, width: int, height: int) -> None:
    """Draw RA/Dec grid lines on the axes with properly angled labels."""
    # Sample grid across image to find coordinate ranges
    grid_density = 100
    y_grid, x_grid = np.mgrid[0:height:grid_density, 0:width:grid_density]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        coords = wcs.pixel_to_world(x_grid.flatten(), y_grid.flatten())
        ra_all = coords.ra.deg
        dec_all = coords.dec.deg

    # Filter valid coordinates
    valid = np.isfinite(ra_all) & np.isfinite(dec_all)
    ra_all = ra_all[valid]
    dec_all = dec_all[valid]

    if len(ra_all) == 0:
        logger.warning("No valid coordinates found for WCS grid")
        return

    # Handle RA wraparound using circular mean
    ra_rad = np.deg2rad(ra_all)
    ra_center = np.rad2deg(np.arctan2(np.mean(np.sin(ra_rad)), np.mean(np.cos(ra_rad))))
    if ra_center < 0:
        ra_center += 360

    # Normalize all RAs around center
    ra_normalized = ra_all - ra_center
    ra_normalized = np.where(ra_normalized > 180, ra_normalized - 360, ra_normalized)
    ra_normalized = np.where(ra_normalized < -180, ra_normalized + 360, ra_normalized)

    # Find percentiles to divide pixels equally (3 lines = 4 bins)
    ra_percentiles = np.percentile(ra_normalized, [25, 50, 75])
    dec_percentiles = np.percentile(dec_all, [25, 50, 75])

    # Convert back to standard RA range
    ra_ticks = ra_percentiles + ra_center
    ra_ticks = np.where(ra_ticks < 0, ra_ticks + 360, ra_ticks)
    ra_ticks = np.where(ra_ticks >= 360, ra_ticks - 360, ra_ticks)

    dec_ticks = dec_percentiles

    # Extend sampling ranges for smooth lines
    dec_range = np.max(dec_all) - np.min(dec_all)
    dec_min_ext = np.min(dec_all) - dec_range
    dec_max_ext = np.max(dec_all) + dec_range

    n_samples = 200
    fs = _font_size(img)
    small_margin = 5  # Just a few pixels to keep anchor inside

    # Draw RA grid lines
    for ra in ra_ticks:
        try:
            dec_samples = np.linspace(dec_min_ext, dec_max_ext, n_samples)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                coords = SkyCoord(
                    ra=np.full(n_samples, ra) * u.deg,
                    dec=dec_samples * u.deg,
                )
                x_coords, y_coords = wcs.world_to_pixel(coords)

            valid = np.isfinite(x_coords) & np.isfinite(y_coords)
            if np.sum(valid) < 2:
                continue

            x_coords = x_coords[valid]
            y_coords = y_coords[valid]

            # Draw line
            ax.plot(x_coords, y_coords, color="white", linestyle="--", alpha=0.7, linewidth=1.5)

            # Find points inside image bounds
            in_bounds = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
            if not np.any(in_bounds):
                continue

            x_in = x_coords[in_bounds]
            y_in = y_coords[in_bounds]

            # Find point closest to bottom edge
            idx = np.argmin(y_in)
            label_x = x_in[idx]
            label_y = y_in[idx]

            # Small nudge to keep anchor just inside
            label_y = max(label_y, small_margin)

            # Calculate angle at label position
            angle = _calculate_line_angle(x_coords, y_coords, label_x, label_y)

            ax.text(
                label_x,
                label_y,
                _format_ra(ra),
                color="white",
                ha="center",
                va="bottom",
                size=fs,
                rotation=angle,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.7, "edgecolor": "none"},
            )
        except Exception as e:
            logger.debug(f"Skipping RA grid line at {ra}: {e}")

    # Draw Dec grid lines
    ra_range = np.max(ra_normalized)
    ra_min_ext = np.min(ra_normalized) - ra_range + ra_center
    ra_max_ext = np.max(ra_normalized) + ra_range + ra_center

    for dec in dec_ticks:
        try:
            ra_samples = np.linspace(ra_min_ext, ra_max_ext, n_samples)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                coords = SkyCoord(
                    ra=ra_samples * u.deg,
                    dec=np.full(n_samples, dec) * u.deg,
                )
                x_coords, y_coords = wcs.world_to_pixel(coords)

            valid = np.isfinite(x_coords) & np.isfinite(y_coords)
            if np.sum(valid) < 2:
                continue

            x_coords = x_coords[valid]
            y_coords = y_coords[valid]

            # Draw line
            ax.plot(x_coords, y_coords, color="white", linestyle="--", alpha=0.7, linewidth=1.5)

            # Find points inside image bounds
            in_bounds = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
            if not np.any(in_bounds):
                continue

            x_in = x_coords[in_bounds]
            y_in = y_coords[in_bounds]

            # Find point closest to left edge
            idx = np.argmin(x_in)
            label_x = x_in[idx]
            label_y = y_in[idx]

            # Small nudge to keep anchor just inside
            label_x = max(label_x, small_margin)

            # Calculate angle at label position
            angle = _calculate_line_angle(x_coords, y_coords, label_x, label_y)

            ax.text(
                label_x,
                label_y,
                _format_dec(dec),
                color="white",
                ha="left",
                va="center",
                size=fs,
                rotation=angle,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.7, "edgecolor": "none"},
            )
        except Exception as e:
            logger.debug(f"Skipping Dec grid line at {dec}: {e}")


def _plot_matched_stars(ax, matched_stars: list, wcs: WCS | None) -> None:
    """Plot matched stars from solve result."""
    if not matched_stars:
        return

    for star in matched_stars:
        # Use x, y if available, otherwise convert from RA/Dec
        if hasattr(star, "x") and hasattr(star, "y") and star.x is not None:
            x, y = star.x, star.y
        elif wcs is not None and hasattr(star, "ra") and hasattr(star, "dec"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                coord = SkyCoord(ra=star.ra * u.deg, dec=star.dec * u.deg)
                x, y = wcs.world_to_pixel(coord)
        else:
            continue

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        # Draw crosshair
        size = 15
        gap = 5
        lw = 1.5

        # Horizontal lines with gap
        ax.plot([x - size, x - gap], [y, y], color="lime", linewidth=lw,
                path_effects=[withStroke(linewidth=lw * 2, foreground="black")])
        ax.plot([x + gap, x + size], [y, y], color="lime", linewidth=lw,
                path_effects=[withStroke(linewidth=lw * 2, foreground="black")])

        # Vertical lines with gap
        ax.plot([x, x], [y - size, y - gap], color="lime", linewidth=lw,
                path_effects=[withStroke(linewidth=lw * 2, foreground="black")])
        ax.plot([x, x], [y + gap, y + size], color="lime", linewidth=lw,
                path_effects=[withStroke(linewidth=lw * 2, foreground="black")])


def _plot_catalog_stars(ax, catalog_stars: list, wcs: WCS | None) -> None:
    """Plot catalog stars as hollow circles."""
    if not catalog_stars or wcs is None:
        return

    from matplotlib.patches import Circle

    for star in catalog_stars:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            coord = SkyCoord(ra=star.ra * u.deg, dec=star.dec * u.deg)
            x, y = wcs.world_to_pixel(coord)

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        # Scale radius by magnitude (brighter = larger)
        # Mag 6 -> radius 20, Mag 16 -> radius 8
        radius = max(8, 25 - star.magnitude)

        circle = Circle(
            (x, y),
            radius=radius,
            facecolor="none",
            edgecolor="red",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.add_patch(circle)
