import rasterio
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from typing import Tuple


def read_image(path):
    with rasterio.open(path) as src:
        image_data = src.read().astype(np.uint16)
        image_data = np.transpose(image_data, (1, 2, 0))  # (height, width, channels)

        return image_data


def read_label(path):
    with rasterio.open(path) as src:
        label_array = src.read(1)

    label_array = label_array.astype(
        np.int16
    )  # Use int16 to initially accommodate -1 values
    label_array[label_array == -1] = 255  # convert -1 to 255

    return label_array.astype(np.uint8)  # Convert to uint8 for TensorFlow processing


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tf_example(
    image_array,
    label_array,
    chip_id,
):
    """
    Converts an image, its label, and metadata (e.g., image ID) to a tf.train.Example message.

    Parameters:
    - image_array: numpy array of the image (uint16).
    - label_array: numpy array of the label (uint8 after remapping).
    - chip_id: an integer representing the image ID.
    """
    assert image_array.dtype == np.uint16
    assert label_array.dtype == np.uint8

    # Convert arrays to bytes
    image_bytes = image_array.tobytes()
    label_bytes = label_array.tobytes()

    # Create a Features message using tf.train.Example
    feature = {
        "image": _bytes_feature(image_bytes),
        "label": _bytes_feature(label_bytes),
        "chip_id": _int64_feature([chip_id]),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def write_tfrecord(tfrecord_filename, split_df, images_path, labels_path):
    """Writes given images and labels to a TFRecord file."""
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for idx, row in split_df.iterrows():
            image_path = os.path.join(
                images_path, row["scene"].replace("S1Hand", "S2Hand")
            )
            label_path = os.path.join(labels_path, row["mask"])

            image_array = read_image(image_path)
            label_array = read_label(label_path)
            chip_id = int(os.path.basename(image_path).split("_")[1])

            example = create_tf_example(image_array, label_array, chip_id)

            writer.write(example)


def get_feature_description() -> dict:
    return {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
        "chip_id": tf.io.FixedLenFeature([1], tf.int64),
    }


def _parse_function(example_proto, feature_description):
    return tf.io.parse_single_example(example_proto, feature_description)


def decode_image(image):
    image = tf.io.decode_raw(image, tf.uint16)
    image = tf.reshape(image, [512, 512, 13])  # Reshape to original shape
    image = image.numpy()
    image = np.transpose(
        image, (2, 0, 1)
    )  # Transpose back to (channels, height, width)

    return image


def decode_label(label):
    label = tf.io.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [512, 512])  # Reshape to original shape
    label = tf.cast(label, tf.int16)  # Cast to int16 to handle negative values
    label = label.numpy()
    label[label == 255] = -1  # convert 255 back to -1

    return label


def decode_chip_id(chip_id):
    return chip_id.numpy()[0]


def parse_tfrecord(tfrecord_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    feature_description = get_feature_description()

    parsed_dataset = raw_dataset.map(
        lambda example_proto: _parse_function(example_proto, feature_description)
    )
    images = []
    labels = []
    chip_ids = []

    for parsed_record in parsed_dataset:
        image_bytes = parsed_record["image"]
        label_bytes = parsed_record["label"]
        chip_id = parsed_record["chip_id"]

        image = decode_image(image_bytes)
        label = decode_label(label_bytes)
        chip_id = decode_chip_id(chip_id)

        images.append(image)
        labels.append(label)
        chip_ids.append(chip_id)

    images = np.array(images)
    labels = np.array(labels)
    chip_ids = np.array(chip_ids)

    return images, labels, chip_ids


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    S2Hand_path = "sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/"
    LabelHand_path = "sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/"
    splits_path = "sen1floods11/v1.1/splits/flood_handlabeled/"

    bolivia_data_path = os.path.join(splits_path, "flood_bolivia_data.csv")

    bolivia_data = pd.read_csv(bolivia_data_path, header=None, names=["scene", "mask"])

    write_tfrecord(
        "bolivia_data.tfrecord",
        split_df=bolivia_data,
        images_path=S2Hand_path,
        labels_path=LabelHand_path,
    )

    raw_dataset = tf.data.TFRecordDataset("bolivia_data.tfrecord")

    feature_description = get_feature_description()

    parsed_dataset = raw_dataset.map(
        lambda example_proto: _parse_function(example_proto, feature_description)
    )

    idx = 1
    parsed_record = next(iter(parsed_dataset.skip(idx)))

    image_bytes = parsed_record["image"]
    label_bytes = parsed_record["label"]
    chip_id = parsed_record["chip_id"]

    image = decode_image(image_bytes)
    label = decode_label(label_bytes)
    chip_id = decode_chip_id(chip_id)

    # Display the image and label
    band_number = 8

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image[(band_number - 1), :, :], cmap="turbo")
    ax[0].set_title("Band {}".format(band_number))
    ax[1].imshow(label, cmap="turbo")
    ax[1].set_title("Label")
    title = f"Decoded Image and Label from TFRecord for Chip ID {chip_id}"
    plt.suptitle(title)
    plt.show()
