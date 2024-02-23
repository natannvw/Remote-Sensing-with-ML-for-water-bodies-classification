import os
import rasterio
import numpy as np
import pandas as pd
from typing import Tuple
import earthpy.spatial as es


def dn2reflectance(
    dn, gain=None, offset=None, satellite="sentinel2", process_lvl="L1C"
) -> np.ndarray:
    if "sentinel2" in satellite:
        if process_lvl == "L1C":
            gain = 0.0001
            offset = 0.0
        else:
            raise ValueError("other process_lvl not implemented yet")

    return dn * gain + offset


def get_satellite_wavelength(satellite) -> dict:
    """
    Get the wavelengths of the satellite bands.

    Parameters
    ----------
    satellite : str
        The satellite name.

    Returns
    -------
    wavelength : dict
        The wavelengths of the satellite bands.

    Example
    -------
    >>> wavelength = satellite_wavelength('sentinel')
    >>> print(wavelength)
    {'B01': 0.4439, 'B02': 0.4966, 'B03': 0.56, 'B04': 0.6645, 'B05': 0.7039, 'B06': 0.7402, 'B07': 0.7825, 'B08': 0.8351, 'B8A': 0.8648, 'B11': 1.6137, 'B12': 2.22024}
    """

    satellite = satellite.lower()

    if "sentinel" in satellite:
        wavelength = {
            "B01": 0.4439,  # Band 1 - Coastal aerosol
            "B02": 0.4966,  # Band 2 - Blue
            "B03": 0.56,  # Band 3 - Green
            "B04": 0.6645,  # Band 4 - Red
            "B05": 0.7039,  # Band 5 - Red Edge 1
            "B06": 0.7402,  # Band 6 - Red Edge 2
            "B07": 0.7825,  # Band 7 - Red Edge 3
            "B08": 0.8351,  # Band 8 - NIR
            "B8A": 0.8648,  # Band 9 - Red Edge 4
            "B09": 0.945,  # Band 9 - Water vapor
            "B10": 1375,  # Band 10 - Cirrus
            "B11": 1.6137,  # Band 11 - SWIR 1
            "B12": 2.22024,  # Band 12 - SWIR 2
        }

    else:
        raise ValueError(f"Satellite {satellite} not implemented yet")

    return wavelength


def get_bands_names(wavelength) -> list:
    return list(wavelength.keys())


def get_scenes_list(dir_path: str) -> Tuple[list, list]:
    scenes_paths = [
        os.path.join(dir_path, filename)
        for filename in os.listdir(dir_path)
        if filename.endswith(".tif")
    ]

    scenes_list = []
    chip_ids = []

    # Read the image data into a list of arrays
    for image_path in scenes_paths:
        with rasterio.open(image_path) as src:
            scenes_list.append(src.read())

            # Extract the chip_id (CHIPID part of the filename)
            chip_id = os.path.basename(image_path).split("_")[1]
            chip_ids.append(chip_id)

    return scenes_list, chip_ids


def get_scenes_arr(dir_path):
    # Convert into (scenes, bands, height, width)

    scenes_list, chip_ids = get_scenes_list(dir_path)

    return np.array(scenes_list), np.array(chip_ids)


def calc_water_probabilities(scenes_masks: np.ndarray) -> np.ndarray:
    water_probabilities = []

    for i in range(scenes_masks.shape[0]):
        labels = scenes_masks[i, 0, :, :]

        valid_pixels = np.sum(labels != -1)  # -1: No Data / Not Valid
        water_pixels = np.sum(labels == 1)

        water_probability = water_pixels / valid_pixels if valid_pixels > 0 else None
        water_probabilities.append(water_probability)

    return np.array(water_probabilities)


def calc_avg_water_probability(
    df: pd.DataFrame, water_probabilities_df: pd.DataFrame
) -> float:
    df["chip_id"] = df["scene"].str.extract("_(\d+)_").astype(int)
    df = df.merge(water_probabilities_df, on="chip_id", how="left")
    avg_water_probability = df["water_probability"].mean()

    return avg_water_probability


def calc_ndwi(
    datacube, satellite, bands_names, nodata=0, Green=None, NIR=None
) -> float:
    """
    Calculate the Normalized Difference Water Index (NDWI)
    For more information see:
    https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
    https://en.wikipedia.org/wiki/Normalized_difference_water_index

    Parameters
    ----------
    datacube : numpy.ndarray
        The datacube of the scene
    satellite : str
        The satellite name
    bands_names : list of str
        The names of the bands in the datacube
    nodata : int, optional
        The nodata value, by default 0
    Green : numpy.ndarray, optional
        The green band, by default None
    NIR : numpy.ndarray, optional
        The near infrared band, by default None

    Returns
    -------
    ndwi : numpy.ndarray
        The NDWI values

    Example
    -------
    >>> import rasterio
    >>> import numpy as np
    >>> raster = rasterio.open('test.tif')
    >>> datacube = raster.read()
    >>> bands_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    >>> ndwi = calc_ndwi(datacube, 'sentinel2', bands_names)
    """

    if (Green is None) or (NIR is None):
        if "sentinel" in satellite:
            Green_band_number = "B03"
            NIR_band_number = "B08"
        else:
            raise ValueError(f"Satellite {satellite} not implemented yet")

        Green_band_index = bands_names.index(Green_band_number)
        NIR_band_index = bands_names.index(NIR_band_number)

        Green = datacube[Green_band_index, :, :]
        NIR = datacube[NIR_band_index, :, :]

    Green_masked = np.ma.masked_where(Green == nodata, Green)
    NIR_masked = np.ma.masked_where(NIR == nodata, NIR)

    ndwi = es.normalized_diff(Green_masked, NIR_masked)

    return ndwi

# from matplotlib import pyplot as plt

# plt.imshow(scenes_list[0][0, :, :], cmap='gray')


# import importlib

# importlib.reload(utils)
