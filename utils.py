import os
import rasterio
import numpy as np


def dn2reflectance(
    dn, gain=None, offset=None, satellite="sentinel2", process_lvl="L1C"
):
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

    return wavelength


def get_bands_names(wavelength) -> list:
    return list(wavelength.keys())


def get_scenes_list(dir_path):
    image_paths = [
        os.path.join(dir_path, filename)
        for filename in os.listdir(dir_path)
        if filename.endswith(".tif")
    ]

    scenes_list = []

    # Read the image data into a list of arrays
    for image_path in image_paths:
        with rasterio.open(image_path) as src:
            scenes_list.append(src.read())

    return scenes_list


def get_scenes_arr(dir_path):
    # Convert into (scenes, bands, height, width)

    scenes_list = get_scenes_list(dir_path)
    return np.array(scenes_list)
