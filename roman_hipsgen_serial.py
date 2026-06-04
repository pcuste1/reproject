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

# ignore warnings from roman datamodels version mismatch
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
    asdf_file = rdm.open(f"https://stpubdata.s3.us-east-1.amazonaws.com/roman/nexus/soc_simulations/r00342/l3/{file_path}_coadd.asdf")
    
    data = np.asarray(asdf_file.data)
    wcs = asdf_file.meta.wcs
    ndd = NDData(data=data, wcs=wcs)

    # tune percentiles as desired
    vmin, vmax = np.nanpercentile(data, [1, 99])  
    
    properties = {"hips_pixel_cut": f"{vmin} {vmax}"}
    
    print(f"Processing {file_path} with vmin={vmin} and vmax={vmax}")
    hips.reproject_to_hips(
        input_data=ndd,
        coord_system_out="equatorial",
        reproject_function=reproject.interpolation.reproject_interp,
        output_directory=output_dir,
        level=13,
        properties=properties,
        progress_bar=tqdm,
        threads=10
    )

def process_l3_from_s3(output_dir):
    logger = logging.getLogger()
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    start_time = datetime.now()
    files = [
        # 'r00342_p_v01001001005001_008m46x51y68_f158',
        'r00342_p_v01001001005001_008m46x52y67_f158',
        'r00342_p_v01001001005001_008m46x52y68_f158',
        'r00342_p_v01001001005001_008m46x53y66_f158',
        'r00342_p_v01001001005001_008m46x53y67_f158',
        'r00342_p_v01001001005001_008m46x53y68_f158',
        'r00342_p_v01001001005001_008m46x54y66_f158',
        'r00342_p_v01001001005001_008m46x54y67_f158',
        'r00342_p_v01001001005001_008m46x54y68_f158',
        'r00342_p_v01001001005001_008m46x55y66_f158',
        'r00342_p_v01001001005001_008m46x55y67_f158',
        'r00342_p_v01001001005001_008m46x55y68_f158',
        'r00342_p_v01001001005001_008m46x56y68_f158',
        'r00342_p_v01001001005001_011m46x48y37_f158',
        'r00342_p_v01001001005001_011m46x48y38_f158',
        'r00342_p_v01001001005001_011m46x49y35_f158',
        'r00342_p_v01001001005001_011m46x49y36_f158',
        'r00342_p_v01001001005001_011m46x49y37_f158',
        'r00342_p_v01001001005001_011m46x49y38_f158',
        'r00342_p_v01001001005001_011m46x49y39_f158',
        'r00342_p_v01001001005001_011m46x49y40_f158',
        'r00342_p_v01001001005001_011m46x50y33_f158',
        'r00342_p_v01001001005001_011m46x50y34_f158',
        'r00342_p_v01001001005001_011m46x50y35_f158',
        'r00342_p_v01001001005001_011m46x50y36_f158',
        'r00342_p_v01001001005001_011m46x50y37_f158',
        'r00342_p_v01001001005001_011m46x50y38_f158',
        'r00342_p_v01001001005001_011m46x50y39_f158',
        'r00342_p_v01001001005001_011m46x50y40_f158',
        'r00342_p_v01001001005001_011m46x51y33_f158',
        'r00342_p_v01001001005001_011m46x51y34_f158',
        'r00342_p_v01001001005001_011m46x51y35_f158',
        'r00342_p_v01001001005001_011m46x51y36_f158',
        'r00342_p_v01001001005001_011m46x51y37_f158',
        'r00342_p_v01001001005001_011m46x51y38_f158',
        'r00342_p_v01001001005001_011m46x51y39_f158',
        'r00342_p_v01001001005001_011m46x51y40_f158',
        'r00342_p_v01001001005001_011m46x51y41_f158',
        'r00342_p_v01001001005001_011m46x52y32_f158',
        'r00342_p_v01001001005001_011m46x52y33_f158',
        'r00342_p_v01001001005001_011m46x52y34_f158',
        'r00342_p_v01001001005001_011m46x52y35_f158',
        'r00342_p_v01001001005001_011m46x52y36_f158',
        'r00342_p_v01001001005001_011m46x52y37_f158',
        'r00342_p_v01001001005001_011m46x52y38_f158',
        'r00342_p_v01001001005001_011m46x52y39_f158',
        'r00342_p_v01001001005001_011m46x52y40_f158',
        'r00342_p_v01001001005001_011m46x52y41_f158',
        'r00342_p_v01001001005001_011m46x52y42_f158',
        'r00342_p_v01001001005001_011m46x53y32_f158',
        'r00342_p_v01001001005001_011m46x53y33_f158',
        'r00342_p_v01001001005001_011m46x53y34_f158',
        'r00342_p_v01001001005001_011m46x53y35_f158',
        'r00342_p_v01001001005001_011m46x53y36_f158',
        'r00342_p_v01001001005001_011m46x53y37_f158',
        'r00342_p_v01001001005001_011m46x53y38_f158',
        'r00342_p_v01001001005001_011m46x53y39_f158',
        'r00342_p_v01001001005001_011m46x53y40_f158',
        'r00342_p_v01001001005001_011m46x53y41_f158',
        'r00342_p_v01001001005001_011m46x53y42_f158',
        'r00342_p_v01001001005001_011m46x54y32_f158',
        'r00342_p_v01001001005001_011m46x54y33_f158',
        'r00342_p_v01001001005001_011m46x54y34_f158',
        'r00342_p_v01001001005001_011m46x54y35_f158',
        'r00342_p_v01001001005001_011m46x54y36_f158',
        'r00342_p_v01001001005001_011m46x54y37_f158',
        'r00342_p_v01001001005001_011m46x54y38_f158',
        'r00342_p_v01001001005001_011m46x54y39_f158',
        'r00342_p_v01001001005001_011m46x54y40_f158',
        'r00342_p_v01001001005001_011m46x54y41_f158',
        'r00342_p_v01001001005001_011m46x54y42_f158',
        'r00342_p_v01001001005001_011m46x54y43_f158',
        'r00342_p_v01001001005001_011m46x55y32_f158',
        'r00342_p_v01001001005001_011m46x55y33_f158',
        'r00342_p_v01001001005001_011m46x55y34_f158',
        'r00342_p_v01001001005001_011m46x55y35_f158',
        'r00342_p_v01001001005001_011m46x55y36_f158',
        'r00342_p_v01001001005001_011m46x55y37_f158',
        'r00342_p_v01001001005001_011m46x55y38_f158',
        'r00342_p_v01001001005001_011m46x55y39_f158',
        'r00342_p_v01001001005001_011m46x55y40_f158',
        'r00342_p_v01001001005001_011m46x55y41_f158',
        'r00342_p_v01001001005001_011m46x55y42_f158',
        'r00342_p_v01001001005001_011m46x55y43_f158',
        'r00342_p_v01001001005001_011m46x56y32_f158',
        'r00342_p_v01001001005001_011m46x56y33_f158',
        'r00342_p_v01001001005001_011m46x56y34_f158',
        'r00342_p_v01001001005001_011m46x56y35_f158',
        'r00342_p_v01001001005001_011m46x56y36_f158',
        'r00342_p_v01001001005001_011m46x56y37_f158',
        'r00342_p_v01001001005001_011m46x56y38_f158',
        'r00342_p_v01001001005001_011m46x56y39_f158',
        'r00342_p_v01001001005001_011m46x56y40_f158',
        'r00342_p_v01001001005001_011m46x56y41_f158',
        'r00342_p_v01001001005001_011m46x56y42_f158',
        'r00342_p_v01001001005001_011m46x56y43_f158',
        'r00342_p_v01001001005001_011m46x57y33_f158',
        'r00342_p_v01001001005001_011m46x57y34_f158',
        'r00342_p_v01001001005001_011m46x57y35_f158',
        'r00342_p_v01001001005001_011m46x57y36_f158'
    ]

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
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ROMAN L3 images and generate HiPS tiles')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='roman_simulation_hips',
        help='Output directory for HiPS tiles (default: roman_simulation_hips)'
    )

    
    args = parser.parse_args()
    output_dir = args.output_dir

    logger = logging.getLogger()

    # Process L3 images from S3 and generate HiPS tiles
    process_l3_from_s3(
        output_dir=output_dir,
    )
