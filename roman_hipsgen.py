import roman_datamodels as rdm
import dask.array as da
import numpy as np
from reproject import hips
import reproject
from reproject.interpolation import reproject_interp
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from asdf.tags.core.ndarray import asdf_datatype_to_numpy_dtype
from astropy.nddata import NDData
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.config import Config

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

from astropy.coordinates import Galactic
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, List, Dict
from threading import Lock
import boto3
import argparse

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(threadName)s] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def process_sky_cell(file_path, output_dir):
    asdf_file = rdm.open(f"https://stpubdata.s3.us-east-1.amazonaws.com/{file_path}")
    
    data = np.asarray(asdf_file.data)
    wcs = asdf_file.meta.wcs
    ndd = NDData(data=data, wcs=wcs)
    
    vmin, vmax = np.nanpercentile(data, [1, 99])   # tune percentiles as desired
    
    properties = {"hips_pixel_cut": f"{vmin} {vmax}"}
    
    hips.reproject_to_hips(
        input_data=ndd,
        coord_system_out="equatorial",
        reproject_function=reproject.interpolation.reproject_interp,
        output_directory=output_dir,
        level=13,
        properties=properties
    )

def process_l3_from_s3(token_holder, bucket_name, prefix, s3_client, output_dir):
    logger = logging.getLogger()
    
    # Get current token (thread-safe)
    with token_holder['lock']:
        current_token = token_holder['token']
    
        # Query S3
        if current_token:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=current_token
            )
        else:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
        
        # Update token for next thread (thread-safe)
        next_token = response.get('NextContinuationToken')
        token_holder['token'] = next_token


    # If no contents, we're done
    if not response.get('Contents'):
        return None

        
    start_time = datetime.now()
    contents = [c for c in response["Contents"] if c["Key"][-10:] == "coadd.asdf"]

    logger.info(f"Beging processing batch with {len(contents)} L3 images.")
    
    count = 0
    success = 0
    # Process your objects here
    for c in contents:
        file_path = c["Key"]
        
        try:
            process_sky_cell(file_path, output_dir)
            success += 1
        except Exception as ex:
            raise ex
        count += 1

        if count % 20 == 0:
            logger.info(f"Processed {count}/{len(contents)} L3 images.")
    
    end_time = datetime.now()
    logger.info(f"[{end_time}] Finished processing batch. {success}/{len(contents)} L3 images processed successfully.")
    return end_time - start_time, count

def process_s3_batches(
    func: Callable,
    bucket_name: str,
    prefix: str,
    num_threads: int = 4,
    s3_client=None,
    **kwargs
) -> List[Any]:
    """
    Process S3 objects with n concurrent threads sharing a continuation token.
    
    Each thread calls func with a thread-safe token holder. The function is
    responsible for querying S3, processing the batch, and updating the token.
    
    Args:
        func: Function called by each thread. Receives (token_holder, bucket_name, 
              prefix, **kwargs). Should:
              1. Get current token from token_holder['token']
              2. Query S3 with that token
              3. Process the batch
              4. Update token_holder['token'] with NextContinuationToken or None
              5. Return result (or None if no more batches)
        bucket_name: S3 bucket name
        prefix: S3 prefix to list objects under
        num_threads: Number of concurrent worker threads
        s3_client: Boto3 S3 client (creates one if not provided)
        **kwargs: Additional keyword arguments to pass to func
    
    Returns:
        List of results from processing each batch
    """
    # Thread-safe token holder
    token_holder = {
        'token': None,  # None means fetch first batch
        'lock': Lock()
    }
    
    results = []
    
    def worker():
        """Worker thread that processes batches using shared token."""
        logger = logging.getLogger()
        logger.info("Thread started, entering worker loop")
        
        iteration = 0
        while True:
            iteration += 1
            logger.debug(f"Iteration {iteration}: Calling process function")
            
            result = func(
                token_holder=token_holder,
                bucket_name=bucket_name,
                prefix=prefix,
                s3_client=s3_client,
                **kwargs
            )
            
            if result is None:
                logger.info("No more batches, thread exiting")
                break
            
            logger.info(f"Batch processed: {result}")
            results.append(result)
    
    logger = logging.getLogger()
    logger.info(f"Starting batch processing with {num_threads} threads")
    
    with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="S3Worker") as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for i, future in enumerate(futures, 1):
            future.result()
            logger.info(f"Thread {i} completed")
    
    logger.info(f"All threads finished. Total results: {len(results)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ROMAN L3 images and generate HiPS tiles')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='roman_simulation_hips',
        help='Output directory for HiPS tiles (default: roman_simulation_hips)'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=4,
        help='Number of threads for concurrent processing (default: 4)'
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir
    num_threads = args.num_threads

    logger = logging.getLogger()
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    results = process_s3_batches(
        func=process_l3_from_s3,
        bucket_name='stpubdata',
        prefix='roman/nexus/soc_simulations/r00342/l3/',
        num_threads=num_threads,
        s3_client=s3,
        output_dir=output_dir
    )

    # Compute lower resolution tiles
    start_time = datetime.now()
    logger.info(f"Starting lower resolution tile computation.")

    hips.compute_lower_resolution_tiles(
        output_directory=output_dir,
        ndim=2,
        frame=Galactic(),
        tile_format="fits",
        tile_size=512,
        tile_depth=16,
        spatial_level=13,
        level_depth=None
    )

    end_time = datetime.now()
    logger.info(f"Finished lower resolution tile computation. Elapsed time: {end_time - start_time}")