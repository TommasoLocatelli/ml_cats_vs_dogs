import numpy as np
import pathlib
import PIL
import PIL.Image
import tensorflow as tf

def ready_to_be_used_dataset(seed=123, image_squared_size=256):
    data_dir = pathlib.Path('downloads\CatsDogs')
    batch_size = 42
    img_height = image_squared_size
    img_width = image_squared_size
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

#https://cloudxlab.com/assessment/displayslide/5658/converting-tensor-to-image
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)