from sklearn.ensemble import RandomForestClassifier
from typing import Literal, Tuple
import os
import numpy as np
import pandas as pd

import tfrecord_utils
import utils
from model_training import train_optimize  # noqa: F401
from itertools import chain, combinations
import mlflow_utils


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


def calc_ndwi_df(df: pd.DataFrame, satellite: str = "sentinel2") -> pd.DataFrame:
    """
    Calculate the Normalized Difference Water Index (NDWI)
    For more information see:
    https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndwi/
    https://en.wikipedia.org/wiki/Normalized_difference_water_index


    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe of the scene
    satellite : str
        The satellite name

    Returns
    -------
    ndvi : pandas.DataFrame
        The NDWI values column

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_csv('test.csv')
    >>> ndvi = calc_ndvi_df(df, 'sentinel2')
    """

    if "sentinel" in satellite:
        Green_band_number = "B03"
        NIR_band_number = "B08"
    else:
        raise ValueError(f"Satellite {satellite} not implemented yet")

    ndwi = (df[Green_band_number] - df[NIR_band_number]) / (
        df[Green_band_number] + df[NIR_band_number]
    )

    return ndwi


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["NDWI"] = calc_ndwi_df(df, satellite="sentinel2")

    return df


def powerset(iterable, include_empty=True):
    s = list(iterable)

    if include_empty:
        n = 0
    else:
        n = 1

    return list(chain.from_iterable(combinations(s, r) for r in range(n, len(s) + 1)))


def get_features_combinations(df: pd.DataFrame) -> list[tuple[str]]:
    combinations = powerset(df.columns, include_empty=False)

    return combinations


def clean_finished_combinations(
    combinations: list[dict], experiment_name: str
) -> list[tuple[list[str]]]:
    finished_configs = mlflow_utils.get_finished_configs(experiment_name)

    skipped_configs = [
        combination for combination in combinations if combination in finished_configs
    ]

    if len(skipped_configs) > 0:
        print(f"Skipped {len(skipped_configs)} configurations")

    # Getting the cleaned combinations that are not in finished_configs
    cleaned_combinations = [
        combination
        for combination in combinations
        if combination not in finished_configs
    ]

    return cleaned_combinations


def ml_pipeline(
    target: str = "label",
    experiment_name: str = "SenFloods",
    retrive_registered_model: bool = False,
) -> RandomForestClassifier:
    # Train model
    train_images, train_labels = get_dataset(data="train")
    valid_images, val_labels = get_dataset(data="valid")

    train_images = utils.dn2reflectance(train_images)
    valid_images = utils.dn2reflectance(valid_images)

    # convert to float32 to avoid overflow
    train_images = train_images.astype(np.float32)
    valid_images = valid_images.astype(np.float32)

    train_df = convert_to_df(train_images, train_labels)
    valid_df = convert_to_df(valid_images, val_labels)

    train_df = feature_engineering(train_df)
    valid_df = feature_engineering(valid_df)

    train_df = train_df.sample(frac=0.1, random_state=42)  # TODO remove this line

    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]

    X_valid = valid_df.drop(target, axis=1)
    y_valid = valid_df[target]

    # best_estimator, best_params, best_score = train_optimize(X_train, y_train)  # Wasn't used

    print("Creating the features combinations...")
    features_combinations = get_features_combinations(X_train)

    print("Starting MLflow server...")
    mlflow_tracking_uri = mlflow_utils.start_mlflow_server()

    experiment_id, mlflow_client = mlflow_utils.set_mlflow(
        experiment_name, mlflow_tracking_uri=mlflow_tracking_uri
    )

    # TODO remove trim (testing purposes)
    features_combinations = features_combinations[:10]

    features_combinations = clean_finished_combinations(
        features_combinations, experiment_name
    )

    print("Trainings runs on MLflow experiment...")


if __name__ == "__main__":
    ml_pipeline()
