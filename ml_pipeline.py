from sklearn.ensemble import RandomForestClassifier
from typing import Literal, Tuple
import os
import numpy as np
import pandas as pd

import tfrecord_utils
import utils


def get_dataset(
    data: Literal["train", "valid", "test", "bolivia"] = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    data_folder = "tfrecord_data"

    match data:
        case "train":
            filename = "train"
        case "valid":
            filename = "valid"
        case "test":
            filename = "test"
        case "bolivia":
            filename = "bolivia"
        case _:
            raise ValueError(f"Data {data} not implemented yet")

    filename = f"{filename}_data.tfrecord"

    path = os.path.join(data_folder, filename)

    images, labels, chip_ids = tfrecord_utils.parse_tfrecord(path)

    return images, labels


def convert_to_df(images, labels):
    n_images, bands, height, width = images.shape

    # train_images_flattened = raster.datacube.reshape(bands, -1).T
    train_images_flattened = images.reshape(-1, bands)
    train_labels_flattened = labels.flatten()

    wavelengths = utils.get_satellite_wavelength("sentinel2")
    bands_names = utils.get_bands_names(wavelengths)

    df = pd.DataFrame(train_images_flattened, columns=bands_names)
    df["label"] = train_labels_flattened

    return df


def ml_pipeline(
    target: str = "water",
    experiment_name: str = "SenFloods",
    retrive_registered_model: bool = False,
) -> RandomForestClassifier:
    # Train model
    train_images, train_labels = get_dataset(data="train")
    valid_images, val_labels = get_dataset(data="valid")

    train_df = convert_to_df(train_images, train_labels)
    valid_df = convert_to_df(valid_images, val_labels)
    model = RandomForestClassifier()


if __name__ == "__main__":
    ml_pipeline()
