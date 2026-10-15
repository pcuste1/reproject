#!/usr/bin/env python
"""Estimate a collection-wide HiPS pixel cut from its highest-order FITS tiles."""

import argparse
import io
import json
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests
from astropy.io import fits
from astropy.stats import mad_std, sigma_clip
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)
DEFAULT_URL = "https://mastimg.stsci.edu/surveys/hips/patrick/reproject_test_6/"
THREAD_LOCAL = threading.local()


def http_session():
    if not hasattr(THREAD_LOCAL, "session"):
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        THREAD_LOCAL.session = session
    return THREAD_LOCAL.session


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=DEFAULT_URL, help="HiPS directory or URL")
    parser.add_argument("--order", type=int, help="Order to analyze; defaults to hips_order")
    parser.add_argument("--sample-size", type=int, default=2_000_000)
    parser.add_argument("--sigma-lower", type=float, default=3.0)
    parser.add_argument("--sigma-upper", type=float, default=5.0)
    parser.add_argument("--maxiters", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-tiles", type=int, help="Analyze only this many tiles (for testing)")
    parser.add_argument("--output", type=Path, default=Path("hips_cut_analysis.json"))
    return parser.parse_args()


def parse_properties(text):
    properties = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key.strip()] = value.strip()
    return properties


def load_properties(source):
    if source.startswith(("http://", "https://")):
        response = http_session().get(
            urljoin(source.rstrip("/") + "/", "properties"), timeout=30
        )
        response.raise_for_status()
        return parse_properties(response.text)
    return parse_properties((Path(source) / "properties").read_text())


def tile_relative_path(order, pixel, extension="fits"):
    directory = pixel // 10_000 * 10_000
    return f"Norder{order}/Dir{directory}/Npix{pixel}.{extension}"


def remote_tile_exists(url):
    response = http_session().head(url, allow_redirects=True, timeout=30)
    if response.status_code == 200:
        return True
    if response.status_code in (403, 404):
        return False
    response.raise_for_status()
    return False


def discover_remote_tiles(source, order, extension, workers, max_tiles=None):
    base_url = source.rstrip("/") + "/"
    pixels = list(range(12))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for level in range(order + 1):
            urls = [urljoin(base_url, tile_relative_path(level, pixel, extension)) for pixel in pixels]
            exists = executor.map(remote_tile_exists, urls)
            pixels = [pixel for pixel, present in zip(pixels, exists, strict=True) if present]
            LOGGER.info("Found %d tiles at order %d", len(pixels), level)
            if not pixels:
                raise RuntimeError(
                    "No remote tiles found; the server may not support HEAD requests. "
                    "Download the collection and pass its local directory instead."
                )
            if max_tiles is not None:
                pixels = pixels[:max_tiles]
            if level < order:
                pixels = [child for pixel in pixels for child in range(4 * pixel, 4 * pixel + 4)]
    return [urljoin(base_url, tile_relative_path(order, pixel, extension)) for pixel in pixels]


def discover_tiles(source, order, extension, workers, max_tiles=None):
    if source.startswith(("http://", "https://")):
        return discover_remote_tiles(source, order, extension, workers, max_tiles)

    directory = Path(source) / f"Norder{order}"
    tiles = sorted(directory.glob(f"Dir*/Npix*.{extension}"))
    if not tiles:
        raise RuntimeError(f"No {extension.upper()} tiles found under {directory}")
    return tiles


def read_finite_pixels(tile):
    if isinstance(tile, str) and tile.startswith(("http://", "https://")):
        response = http_session().get(tile, timeout=120)
        response.raise_for_status()
        source = io.BytesIO(response.content)
    else:
        source = tile

    with fits.open(source, memmap=False) as hdus:
        data = np.asarray(hdus[0].data, dtype=np.float64).ravel()
    return data[np.isfinite(data)]


def add_to_reservoir(values, reservoir, seen, sample_size, rng):
    """Add values using a vectorized form of classic reservoir sampling."""
    fill = min(sample_size - reservoir.size, values.size)
    if fill:
        reservoir = np.concatenate((reservoir, values[:fill]))
        values = values[fill:]
        seen += fill
    if not values.size:
        return reservoir, seen

    previous_seen = seen
    stream_positions = previous_seen + np.arange(1, values.size + 1, dtype=np.int64)
    replacement_indices = rng.integers(stream_positions)
    accepted_positions = np.flatnonzero(replacement_indices < sample_size)
    if accepted_positions.size:
        reversed_positions = accepted_positions[::-1]
        _, last_occurrences = np.unique(
            replacement_indices[reversed_positions], return_index=True
        )
        selected_positions = reversed_positions[last_occurrences]
        reservoir[replacement_indices[selected_positions]] = values[selected_positions]
    return reservoir, previous_seen + values.size


def sample_tiles(tiles, sample_size, workers, seed):
    rng = np.random.default_rng(seed)
    reservoir_values = np.empty(0, dtype=np.float64)
    finite_pixels = 0
    failed_tiles = []
    tile_iterator = iter(tiles)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for tile in tile_iterator:
            pending.append((tile, executor.submit(read_finite_pixels, tile)))
            if len(pending) == 2 * workers:
                break

        while pending:
            tile, future = pending.popleft()
            try:
                values = future.result()
                reservoir_values, finite_pixels = add_to_reservoir(
                    values, reservoir_values, finite_pixels, sample_size, rng
                )
            except Exception as error:
                LOGGER.warning("Could not read %s: %s", tile, error)
                failed_tiles.append(str(tile))

            try:
                next_tile = next(tile_iterator)
            except StopIteration:
                continue
            pending.append((next_tile, executor.submit(read_finite_pixels, next_tile)))

    if not reservoir_values.size:
        raise RuntimeError("No finite pixels were read")
    return reservoir_values, finite_pixels, failed_tiles


def analyze(sample, sigma_lower, sigma_upper, maxiters):
    clipped, lower_bound, upper_bound = sigma_clip(
        sample,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        cenfunc="median",
        stdfunc=mad_std,
        masked=True,
        return_bounds=True,
    )
    retained = clipped.compressed()
    percentiles = np.percentile(sample, (0.5, 1, 50, 99, 99.5))
    return {
        "sigma_clipped_cut": [float(lower_bound), float(upper_bound)],
        "hips_pixel_cut": f"{lower_bound:.17g} {upper_bound:.17g}",
        "sample_percentiles": {
            name: float(value)
            for name, value in zip(("p0.5", "p1", "p50", "p99", "p99.5"), percentiles, strict=True)
        },
        "clipped_median": float(np.median(retained)),
        "clipped_mad_std": float(mad_std(retained)),
        "sample_pixels": int(sample.size),
        "retained_pixels": int(retained.size),
        "rejected_fraction": float(1 - retained.size / sample.size),
    }


def main():
    arguments = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if arguments.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if arguments.workers <= 0:
        raise ValueError("--workers must be positive")

    properties = load_properties(arguments.source)
    order = arguments.order if arguments.order is not None else int(properties["hips_order"])
    formats = properties.get("hips_tile_format", "fits").split()
    if "fits" not in formats:
        raise ValueError("The HiPS collection does not advertise FITS tiles")

    tiles = discover_tiles(
        arguments.source, order, "fits", arguments.workers, arguments.max_tiles
    )
    if arguments.max_tiles is not None:
        tiles = tiles[: arguments.max_tiles]
    LOGGER.info("Analyzing %d order-%d tiles", len(tiles), order)

    sample, finite_pixels, failed_tiles = sample_tiles(
        tiles, arguments.sample_size, arguments.workers, arguments.seed
    )
    report = analyze(sample, arguments.sigma_lower, arguments.sigma_upper, arguments.maxiters)
    report.update(
        {
            "source": arguments.source,
            "order": order,
            "tiles_analyzed": len(tiles) - len(failed_tiles),
            "failed_tiles": failed_tiles,
            "finite_pixels_seen": int(finite_pixels),
            "existing_hips_pixel_cut": properties.get("hips_pixel_cut"),
            "sigma_lower": arguments.sigma_lower,
            "sigma_upper": arguments.sigma_upper,
            "maxiters": arguments.maxiters,
            "seed": arguments.seed,
        }
    )
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("Wrote %s", arguments.output)
    print(f"hips_pixel_cut = {report['hips_pixel_cut']}")


if __name__ == "__main__":
    main()