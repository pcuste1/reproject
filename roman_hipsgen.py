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
from astroquery.mast.missions import MastMissions
from astropy.coordinates import SkyCoord
import astropy.units as u
from glob import glob

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

from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(processName)s] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def process_sky_cell(file_path, output_dir):
    asdf_file = rdm.open(f"https://stpubdata.s3.us-east-1.amazonaws.com/roman/nexus/soc_simulations/r00342/l3/{file_path}")
    
    data = np.asarray(asdf_file.data)
    # data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    wcs = asdf_file.meta.wcs
    ndd = NDData(data=data, wcs=wcs)

    vmin, vmax = np.nanpercentile(data, [1, 99])   # tune percentiles as desired
    
    properties = {"hips_pixel_cut": f"{vmin} {vmax}"}
    
    hips.reproject_to_hips(
        input_data=ndd,
        coord_system_out="equatorial",
        reproject_function=reproject.interpolation.reproject_interp,
        # reproject_function=reproject.reproject_adaptive,
        output_directory=output_dir,
        level=13,
        properties=properties,
        # progress_bar=tqdm,
        threads=10
    )

# Use MAST test to access simulated Roman observations:
def _use_mast_test(mast):
    mast._service_api_connection.SERVICE_URL = 'https://mastint.stsci.edu'
    mast._service_api_connection.REQUEST_URL = 'https://mastint.stsci.edu/search/roman/api/v0.1/'
    mast._service_api_connection.MISSIONS_DOWNLOAD_URL = 'https://mastint.stsci.edu/search/'

def process_l3_from_s3(shared_data, bucket_name, prefix, output_dir):
    logger = logging.getLogger()
    
    # Create S3 client in the worker process
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Get current token (process-safe)
    with shared_data['lock']:
        current_offset = shared_data['offset']

        mm = MastMissions(mission='roman')
        _use_mast_test(mm)
    
        # Query missions mast
        #8.42377562 -44.6996920
        # coord = SkyCoord(ra=8.42377562 * u.deg, dec=-44.6996920 * u.deg)
        query_results = mm.query_criteria(
            # coordinates=coord, 
            # radius=30*u.arcmin, 
            productLevel='3',
            offset=current_offset,
            limit=10,
            optical_element='F158',
            program='00342',
            select_cols=['fileSetName']
        )
        
        # Update token for next process (process-safe)
        shared_data['offset'] = current_offset + len(query_results)

    # If no contents, we're done
    if not query_results:
        return None

    start_time = datetime.now()
    files = [f"{obs['fileSetName']}_coadd.asdf" for obs in query_results]

    logger.info(f"Begin processing batch with {len(files)} L3 images.")
    
    count = 0
    success = 0
    # Process your objects here
    for file in files:
        try:
            process_sky_cell(file, output_dir)
            success += 1
        except Exception as ex:
            logger.error(f"Error processing {file}: {ex}")
        count += 1

        if count % 20 == 0:
            logger.info(f"Processed {count}/{len(files)} L3 images.")
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info(f"[{elapsed_time}] Finished processing batch. {success}/{len(files)} L3 images processed successfully.")
    return elapsed_time, count

def _worker_process(shared_data, bucket_name, prefix, max_files, func, kwargs):
    """Module-level worker function for pickling compatibility."""
    logger = logging.getLogger()
    logger.info("Process started, entering worker loop")
    
    iteration = 0
    total_files_processed = 0
    while iteration < max_files/10:  # Prevent infinite loop
        iteration += 1
        logger.debug(f"Iteration {iteration}: Calling process function")
        
        elapsed_time, processed_file_count = func(
            shared_data=shared_data,
            bucket_name=bucket_name,
            prefix=prefix,
            **kwargs
        )
        
        if processed_file_count == 0:
            logger.info("No more batches, process exiting")
            break
        
        logger.info(f"Batch processed: {elapsed_time}, {processed_file_count}")
        total_files_processed += processed_file_count
    
    return total_files_processed

def process_s3_batches(
    func: Callable,
    bucket_name: str,
    prefix: str,
    num_processes: int = 4,
    max_files: int = 100,
    offset: int = 0,
    **kwargs
) -> List[Any]:
    """
    Process S3 objects with n concurrent processes sharing a continuation token.
    
    Each process calls func with a process-safe token holder. The function is
    responsible for querying S3, processing the batch, and updating the token.
    
    Args:
        func: Function called by each process. Receives (shared_data, bucket_name, 
              prefix, **kwargs). Should:
              1. Get current token from shared_data['token']
              2. Query S3 with that token
              3. Process the batch
              4. Update shared_data['token'] with NextContinuationToken or None
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
        shared_data = manager.dict()
        shared_data['offset'] = offset
        shared_data['lock'] = manager.Lock()
        
        logger = logging.getLogger()
        logger.info(f"Starting batch processing with {num_processes} processes")
        
        all_results = 0
        process_max_files = max_files / num_processes
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(_worker_process, shared_data, bucket_name, prefix, process_max_files, func, kwargs)
                for _ in range(num_processes)
            ]
            for i, future in enumerate(futures, 1):
                process_results = future.result()
                all_results += process_results
                logger.info(f"Process {i} completed processing {process_results} files")
        
        logger.info(f"All processes finished. Total files processed: {all_results}")
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
    parser.add_argument(
        '--max-files',
        type=int,
        default=100,
        help='Maximum number of files to process (default: 100)'
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Offset for processing (default: 0)'
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['process_l3', 'compute_tiles', 'both'],
        default='both',
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    num_processes = args.num_processes
    max_files = args.max_files
    offset = args.offset

    logger = logging.getLogger()

    if args.action in ['process_l3', 'both']:
        # Process L3 images from S3 and generate HiPS tiles
        results = process_s3_batches(
            func=process_l3_from_s3,
            bucket_name='stpubdata',
            prefix='roman/nexus/soc_simulations/r00342/l3/',
            num_processes=num_processes,
            output_dir=output_dir,
            max_files=max_files,
            offset=offset
        )

    if args.action in ['compute_tiles', 'both']:
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