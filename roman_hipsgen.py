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
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Any, List, Dict
from multiprocessing import Manager
import boto3
import argparse

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(processName)s] %(asctime)s - %(levelname)s - %(message)s',
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

def process_l3_from_s3(token_holder, bucket_name, prefix, output_dir):
    logger = logging.getLogger()
    
    # Create S3 client in the worker process
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Get current token (process-safe)
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
        
        # Update token for next process (process-safe)
        next_token = response.get('NextContinuationToken')
        token_holder['token'] = next_token


    # If no contents, we're done
    if not response.get('Contents'):
        return None

        
    start_time = datetime.now()
    contents = [c for c in response["Contents"] if c["Key"][-10:] == "coadd.asdf"]

    logger.info(f"Begin processing batch with {len(contents)} L3 images.")
    
    count = 0
    success = 0
    # Process your objects here
    for c in contents:
        file_path = c["Key"]
        
        try:
            process_sky_cell(file_path, output_dir)
            success += 1
        except Exception as ex:
            logger.error(f"Error processing {file_path}: {ex}")
        count += 1

        if count % 20 == 0:
            logger.info(f"Processed {count}/{len(contents)} L3 images.")
    
    end_time = datetime.now()
    logger.info(f"[{end_time}] Finished processing batch. {success}/{len(contents)} L3 images processed successfully.")
    return end_time - start_time, count

def _worker_process(token_holder, bucket_name, prefix, func, kwargs):
    """Module-level worker function for pickling compatibility."""
    logger = logging.getLogger()
    logger.info("Process started, entering worker loop")
    
    iteration = 0
    results = []
    while True:
        iteration += 1
        logger.debug(f"Iteration {iteration}: Calling process function")
        
        result = func(
            token_holder=token_holder,
            bucket_name=bucket_name,
            prefix=prefix,
            **kwargs
        )
        
        if result is None:
            logger.info("No more batches, process exiting")
            break
        
        logger.info(f"Batch processed: {result}")
        results.append(result)
    
    return results

def process_s3_batches(
    func: Callable,
    bucket_name: str,
    prefix: str,
    num_processes: int = 4,
    **kwargs
) -> List[Any]:
    """
    Process S3 objects with n concurrent processes sharing a continuation token.
    
    Each process calls func with a process-safe token holder. The function is
    responsible for querying S3, processing the batch, and updating the token.
    
    Args:
        func: Function called by each process. Receives (token_holder, bucket_name, 
              prefix, **kwargs). Should:
              1. Get current token from token_holder['token']
              2. Query S3 with that token
              3. Process the batch
              4. Update token_holder['token'] with NextContinuationToken or None
              5. Return result (or None if no more batches)
        bucket_name: S3 bucket name
        prefix: S3 prefix to list objects under
        num_processes: Number of concurrent worker processes
        **kwargs: Additional keyword arguments to pass to func
    
    Returns:
        List of results from processing each batch
    """
    # Process-safe token holder using Manager
    with Manager() as manager:
        token_holder = manager.dict()
        token_holder['token'] = None  # None means fetch first batch
        token_holder['lock'] = manager.Lock()
        
        logger = logging.getLogger()
        logger.info(f"Starting batch processing with {num_processes} processes")
        
        all_results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(_worker_process, token_holder, bucket_name, prefix, func, kwargs)
                for _ in range(num_processes)
            ]
            for i, future in enumerate(futures, 1):
                process_results = future.result()
                all_results.extend(process_results)
                logger.info(f"Process {i} completed with {len(process_results)} results")
        
        logger.info(f"All processes finished. Total results: {len(all_results)}")
        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ROMAN L3 images and generate HiPS tiles')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='roman_simulation_hips',
        help='Output directory for HiPS tiles (default: roman_simulation_hips)'
    )
    parser.add_argument(
        '--num-processes',
        type=int,
        default=4,
        help='Number of processes for concurrent processing (default: 4)'
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir
    num_processes = args.num_processes

    logger = logging.getLogger()

    results = process_s3_batches(
        func=process_l3_from_s3,
        bucket_name='stpubdata',
        prefix='roman/nexus/soc_simulations/r00342/l3/',
        num_processes=num_processes,
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