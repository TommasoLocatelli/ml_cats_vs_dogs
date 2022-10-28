import numpy as np
import pathlib
import PIL
import PIL.Image
import tensorflow as tf


def ready_to_be_used_dataset(
    seed=123, image_squared_size=256, color_mode="rgb"
):  # color_mode	One of "grayscale", "rgb", "rgba". Default: "rgb"
    data_dir = pathlib.Path("downloads\CatsDogs")
    batch_size = 42
    img_height = image_squared_size
    img_width = image_squared_size
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        color_mode=color_mode,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        color_mode=color_mode,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_val = val_ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_train, normalized_val


# https://cloudxlab.com/assessment/displayslide/5658/converting-tensor-to-image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def performance_plot(history):
    plt.figure(figsize=(16, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss', size=12)
    plt.xlabel('epoch', size=12)
    plt.legend(['train', 'val'], fontsize=10)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy', size=12)
    plt.xlabel('epoch', size=12)
    plt.legend(['train', 'val'], fontsize=10)

    plt.show()
