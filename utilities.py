import numpy as np
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *


def ready_to_be_used_dataset(
    seed=42,
    image_squared_size=256,
    color_mode="rgb",
    batch_size=32,
):  # color_mode	One of "grayscale", "rgb", "rgba". Default: "rgb"

    data_dir = pathlib.Path("downloads\CatsDogs")
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

    normalization_layer = Rescaling(1.0 / 255)
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
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("loss", size=12)
    plt.xlabel("epoch", size=12)
    plt.legend(["train", "val"], fontsize=10)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.ylabel("accuracy", size=12)
    plt.xlabel("epoch", size=12)
    plt.legend(["train", "val"], fontsize=10)

    plt.show()


def delete_from_list(filename="files.to.delete.txt"):
    count = 0
    with open(filename, "rt") as f:
        lines = f.readlines()
        for pth in lines:
            try:
                os.remove(pth.strip())
                count += 1
            except:
                pass

    print(
        count,
        "of",
        len(lines),
        "files deleted",
    )


def five_fold_cross_validation(Model, ds, k=5, no_epochs=5):
    """
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
    """

    inputs = np.array([i[0] for _, (i, l) in enumerate(ds)])
    targets = np.array([l[0] for _, (i, l) in enumerate(ds)])

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):

        # Define the model architecture
        model = Model

        # Compile the model
        model.compile(
            loss=SparseCategoricalCrossentropy(), optimizer=Adam(), metrics=["accuracy"]
        )

        # Generate a print
        print(
            "------------------------------------------------------------------------"
        )
        print(f"Training for fold {fold_no} ...")

        # Fit data to model
        history = model.fit(
            inputs[train], targets[train], batch_size=1, epochs=no_epochs, verbose=1
        )

        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        print(
            f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
        )
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
