import numpy as np
import pathlib
import PIL
import PIL.Image
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from numba import cuda

from pathlib import Path, PurePath

import tensorflow as tf
from tensorflow import keras

from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *
import CNNModels

DATASET_DIRECTORY = pathlib.Path("downloads", "CatsDogs")
import gc


def ready_to_be_used_dataset(
    image_size,
    color_mode="rgb",
    batch_size=32,
    seed=42,
):
    """
    Normalize and split in half `DATASET_DIRECTORY` images.
    """

    training_dataset = image_dataset_from_directory(
        DATASET_DIRECTORY,
        validation_split=0.5,
        color_mode=color_mode,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )

    validation_dataset = image_dataset_from_directory(
        DATASET_DIRECTORY,
        color_mode=color_mode,
        validation_split=0.5,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )

    # normalization_layer = Rescaling(1.0 / 255)
    # normalized_training = training_dataset.map(lambda x, y: (normalization_layer(x), y))
    # normalized_validation = validation_dataset.map(
    #     lambda x, y: (normalization_layer(x), y)
    # )
    # return normalized_training, normalized_validation
    return training_dataset, validation_dataset


# https://cloudxlab.com/assessment/displayslide/5658/converting-tensor-to-image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# plot_model(
#     model,
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
#     layer_range=None,
#     show_layer_activations=True,
# )


def performance_plot(results, tag: str = None, figsize=(16, 6)):
    plt.figure(figsize=figsize)

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(results.history["loss"])
    plt.plot(results.history["val_loss"])
    plt.ylabel("loss", size=12)
    plt.xlabel("epoch", size=12)
    plt.legend(["train", "val"], fontsize=10)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results.history["accuracy"])
    plt.plot(results.history["val_accuracy"])
    plt.ylabel("accuracy", size=12)
    plt.xlabel("epoch", size=12)
    plt.legend(["train", "val"], fontsize=10)

    if tag is not None:
        plt.savefig(PurePath(tag, "loss_accuracy.png"))
    else:
        plt.show()


def delete_from_list(filename="files.to.delete.txt"):
    count = 0
    with open(filename, "rt") as f:
        lines = f.readlines()
        for pth in lines:
            try:
                os.remove(pth.strip())
                count += 1
            except Exception as inst:
                print(inst.args)

    print(
        count,
        "of",
        len(lines),
        "files deleted",
    )

def validate_tag(model, color_mode, image_size):
    tag = f"{model}_model_{color_mode}_x{image_size}_img"
    Path(tag).mkdir(parents=True, exist_ok=True)
    return tag


def auto_train(
    model_name,
    image_size=128,
    color_mode="rgb",
    epochs=24,
    batch_size=32,
):
    # REF TAG
    tag = validate_tag(model_name, image_size=image_size, color_mode=color_mode)

    # DATA
    train_val, test = ready_to_be_used_dataset(
        image_size=image_size,
        color_mode=color_mode,
        batch_size=batch_size,
    )
    train_size = int(len(train_val) * 0.8)
    train, valid = train_val.take(train_size), train_val.skip(train_size)

    # CREATE MODEL
    model = CNNModels.get_model(model_name)
    model.compile(
        optimizer=Adam(),  # NOTE learning rate?
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # FITTING
    results = model.fit(
        x=train,
        validation_data=valid,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.02, patience=3)],
    )

    performance_plot(results, tag)

    # SAVE MODEL
    with open(PurePath(tag, "model_summary.txt"), "wt") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # THAT's ALL FOLKS
    tf.keras.backend.clear_session()
    # cuda.select_device(0)
    # cuda.close()
    # print("GPU MEM RESET")
    del model
    gc.collect()

    return tag, test


def k_fold_cross_validation(
    tag,
    dataset,
    model_name,
    k=5,
    epochs=24,
    batch_size=32,
):
    """
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
    """

    # # NOTE that's gonna cost you A LOT OF RAM
    # inputs = np.array([e for batch, _ in iter(dataset) for e in batch])
    # targets = np.array([l for _, labels in iter(dataset) for l in labels])

    # per-fold score containers
    accuracies = []
    losses = []

    # kfold = KFold(n_splits=k, shuffle=True)
    # fold_counter = 0

    for fold_counter in range(k):

        # ROTATE SPLITTING OF TESTING
        test_size = len(dataset) // k
        train_lx_size = len(dataset) * fold_counter // k

        train_lx = dataset.take(train_lx_size)
        test = dataset.skip(train_lx_size).take(test_size)
        validation = dataset.skip(train_lx_size + test_size).take(test_size)
        train_rx = dataset.skip(train_lx_size + test_size + test_size)

        train = train_rx.concatenate(train_lx)

        # CREATE MODEL
        model = CNNModels.get_model(model_name)
        model.compile(
            optimizer=Adam(),  # NOTE learning rate?
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # FITTING
        print("-" * 35)
        print(f"Training for fold {fold_counter+1} ...")
        print("-" * 35)

        model.fit(
            x=train,
            validation_data=validation,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.02, patience=3)],
            verbose=2,
        )

        # EVALUATING
        print("-" * 35)
        print(f"Testing for fold {fold_counter+1} ...")
        print("-" * 35)

        scores = model.evaluate(test)

        print(f"Score for fold {fold_counter+1}:")
        for i, s in enumerate(model.metrics_names):
            print(f">> {s} of {scores[i]}")

        # SAVE STUFF
        accuracies.append(scores[1])
        losses.append(scores[0])

        # fold_counter += 1

        # THAT's ALL FOLKS
        tf.keras.backend.clear_session()
        # cuda.select_device(0)
        # cuda.close()
        # print("GPU MEM RESET")

        del model
        gc.collect()

    with open(f"accuracies_{tag}.txt", "wt") as f:
        for e in accuracies:
            f.writeline(e)

    return accuracies
