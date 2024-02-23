import rasterio
import numpy as np
import tensorflow as tf
import os


def read_image(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.uint16)


def read_label(path):
    with rasterio.open(path) as src:
        label_array = src.read(1)

    label_array = label_array.astype(
        np.int16
    )  # Use int16 to initially accommodate -1 values
    label_array[label_array == -1] = 255  # convert -1 to 255

    return label_array.astype(np.uint8)  # Convert to uint8 for TensorFlow processing


image_path = r"C:\Users\NathanWeinstein\Desktop\Personal\G\SenFloods\sen1floods11\v1.1\data\flood_events\HandLabeled\S2Hand\Bolivia_23014_S2Hand.tif"
label_path = r"C:\Users\NathanWeinstein\Desktop\Personal\G\SenFloods\sen1floods11\v1.1\data\flood_events\HandLabeled\LabelHand\Bolivia_23014_LabelHand.tif"

image_array = read_image(image_path)
label_array = read_label(label_path)
chip_id = int(os.path.basename(image_path).split("_")[1])


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_example(image_array, label_array, chip_id):
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


# create_example(image_array, label_array, chip_id)


def write_tfrecord(tfrecord_filename, images, labels):
    """Writes given images and labels to a TFRecord file."""
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for image, label in zip(images, labels):
            example = create_example(image, label)
            writer.write(example)


# Example usage
# images and labels should be loaded as numpy arrays beforehand
# write_tfrecord('dataset.tfrecord', images, labels)
