#!/usr/bin/env python
"""Measure Roman ASDF image consistency before generating a HiPS dataset."""

import argparse
import csv
import json
import logging
from pathlib import Path

import boto3
import numpy as np
import roman_datamodels as rdm
from botocore import UNSIGNED
from botocore.config import Config


LOGGER = logging.getLogger(__name__)
METRICS = ("mean", "median", "std", "p01", "p99")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="stpubdata", help="S3 bucket containing the ASDF files")
    parser.add_argument(
        "--prefix",
        default="roman/nexus/soc_simulations/r00342/l3/",
        help="S3 prefix containing the ASDF files",
    )
    parser.add_argument("--output", type=Path, default=Path("asdf_analysis"))
    parser.add_argument("--max-files", type=int, default=100, help="Analyze at most this many files")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1_000_000,
        help="Maximum evenly-spaced pixels per image; use 0 for all pixels",
    )
    parser.add_argument("--full-data", action="store_true", help="Read every pixel from every file")
    parser.add_argument("--suffix", default="_coadd.asdf", help="Only analyze keys ending in this suffix")
    return parser.parse_args()


def list_asdf_keys(s3_client, bucket, prefix, suffix, max_files):
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(suffix):
                yield key
                count += 1
                if max_files is not None and count >= max_files:
                    return


def sample_data(data, max_samples):
    """Return a deterministic spatial sample without flattening a lazy ASDF array."""
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D image, found shape {data.shape}")
    if not max_samples or data.size <= max_samples:
        return np.asarray(data, dtype=np.float64), 1

    stride = int(np.ceil(np.sqrt(data.size / max_samples)))
    return np.asarray(data[::stride, ::stride], dtype=np.float64), stride


def file_statistics(key, bucket, max_samples):
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    model = rdm.open(url)
    try:
        image_shape = model.data.shape
        sample, stride = sample_data(model.data, max_samples)
    finally:
        model.close()

    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        raise ValueError("image contains no finite pixels")

    p01, p05, p25, median, p75, p95, p99, p999 = np.percentile(
        finite, (1, 5, 25, 50, 75, 95, 99, 99.9)
    )
    return {
        "key": key,
        "image_shape": "x".join(map(str, image_shape)),
        "sample_shape": "x".join(map(str, sample.shape)),
        "sample_stride": stride,
        "sample_pixels": int(sample.size),
        "finite_pixels": int(finite.size),
        "finite_fraction": float(finite.size / sample.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(median),
        "std": float(np.std(finite)),
        "mad": float(np.median(np.abs(finite - median))),
        "p01": float(p01),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "p99": float(p99),
        "p999": float(p999),
        "dynamic_range_p01_p99": float(p99 - p01),
        "sum": float(np.sum(finite)),
        "sum_squares": float(np.sum(finite * finite)),
    }


def robust_z_scores(values):
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return median, mad, np.zeros_like(values)
    return median, mad, 0.6745 * (values - median) / mad


def summarize(rows):
    finite_pixels = sum(row["finite_pixels"] for row in rows)
    pixel_sum = sum(row["sum"] for row in rows)
    pixel_sum_squares = sum(row["sum_squares"] for row in rows)
    mean = pixel_sum / finite_pixels
    return {
        "files_analyzed": len(rows),
        "sampled_finite_pixels": finite_pixels,
        "sampled_pixel_mean": mean,
        "sampled_pixel_std": float(np.sqrt(pixel_sum_squares / finite_pixels - mean * mean)),
        "sampled_min": min(row["min"] for row in rows),
        "sampled_max": max(row["max"] for row in rows),
        "recommended_hips_pixel_cut": [
            float(np.median([row["p01"] for row in rows])),
            float(np.median([row["p99"] for row in rows])),
        ],
    }


def add_normalization_flags(rows):
    reference = {}
    for metric in METRICS:
        reference[metric], _, scores = robust_z_scores([row[metric] for row in rows])
        for row, score in zip(rows, scores, strict=True):
            row[f"{metric}_robust_z"] = float(score)

    reference_range = reference["p99"] - reference["p01"]
    for row in rows:
        row["suggested_background_offset"] = float(reference["median"] - row["median"])
        data_range = row["dynamic_range_p01_p99"]
        row["suggested_scale_factor"] = float(reference_range / data_range) if data_range else None
        row["normalization_flag"] = any(
            abs(row[f"{metric}_robust_z"]) > 3.5 for metric in METRICS
        ) or row["finite_fraction"] < 0.95
    return reference


def write_reports(output, rows, summary, reference):
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row.pop("sum")
        row.pop("sum_squares")

    with (output / "per_file_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flagged = [row["key"] for row in rows if row["normalization_flag"]]
    report = {
        "summary": summary,
        "reference_file_statistics": reference,
        "normalization_threshold": "absolute robust z-score > 3.5 or finite fraction < 0.95",
        "files_requiring_review": flagged,
    }
    with (output / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)


def main():
    arguments = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if arguments.max_samples < 0:
        raise ValueError("--max-samples must be zero or positive")

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    rows = []
    for number, key in enumerate(
        list_asdf_keys(
            s3_client, arguments.bucket, arguments.prefix, arguments.suffix, arguments.max_files
        ),
        start=1,
    ):
        try:
            rows.append(file_statistics(key, arguments.bucket, 0 if arguments.full_data else arguments.max_samples))
            LOGGER.info("Analyzed %d: %s", number, key)
        except Exception as error:
            LOGGER.exception("Skipping %s: %s", key, error)

    if not rows:
        raise RuntimeError("No ASDF files were successfully analyzed")
    summary = summarize(rows)
    reference = add_normalization_flags(rows)
    write_reports(arguments.output, rows, summary, reference)
    LOGGER.info("Wrote %d file statistics to %s", len(rows), arguments.output)
    LOGGER.info("Recommended shared hips_pixel_cut: %s", summary["recommended_hips_pixel_cut"])


if __name__ == "__main__":
    main()