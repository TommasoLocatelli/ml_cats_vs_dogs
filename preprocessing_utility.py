import numpy as np
import pathlib
import PIL
import PIL.Image
import tensorflow as tf

def ready_to_be_used_dataset(seed=123):
    data_dir = pathlib.Path('downloads\CatsDogs')
    batch_size = 42
    img_height = 1000
    img_width = 1000
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val = val_ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_train, normalized_val