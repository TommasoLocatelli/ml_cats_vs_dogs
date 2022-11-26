import numpy as np
import pathlib
import PIL
import PIL.Image
import os
import gc
from matplotlib import pyplot as plt
from pathlib import Path, PurePath

import tensorflow as tf

from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *
import CNNModels

DATASET_DIRECTORY = pathlib.Path("downloads", "CatsDogs")


def ready_to_be_used_dataset(
    image_size,
    color_mode="rgb",
    seed=42,
):
    """
    Normalize and split in half `DATASET_DIRECTORY` images.
    """

    training_dataset = image_dataset_from_directory(
        DATASET_DIRECTORY,
        validation_split=0.7,
        color_mode=color_mode,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
    )

    validation_dataset = image_dataset_from_directory(
        DATASET_DIRECTORY,
        color_mode=color_mode,
        validation_split=0.7,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
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
):
    # REF TAG
    tag = validate_tag(model_name, image_size=image_size, color_mode=color_mode)

    # DATA
    train_val, test = ready_to_be_used_dataset(
        image_size=image_size,
        color_mode=color_mode,
    )

    valid_size = len(train_val) // 5
    valid, train = train_val.take(valid_size), train_val.skip(valid_size)

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
        callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=4)],
    )

    performance_plot(results, tag)

    # SAVE MODEL
    with open(PurePath(tag, "model_summary.txt"), "wt") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # THAT's ALL FOLKS
    del model
    gc.collect()

    return tag, test


def k_fold_cross_validation(
    tag,
    dataset,
    model_name,
    k=5,
    epochs=24,
    learning_rate=1e-3,
):
    """
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
    """

    # per-fold score containers
    accuracies = []
    losses = []

    for fold_counter in range(k):

        # ROTATE SPLITTING OF TESTING
        N = len(dataset)
        fold_size =  N // k
        train_lx_size = N * fold_counter // k

        train_lx = dataset.take(train_lx_size)
        test = dataset.skip(train_lx_size).take(fold_size)
        validation = dataset.skip(train_lx_size + fold_size).take(fold_size)
        train_rx = dataset.skip(train_lx_size + fold_size + fold_size)

        if fold_counter == k-1:
            validation, train_lx = train_lx.take(fold_size), train_lx.skip(fold_size)

        train = train_rx.concatenate(train_lx)

        print("FSIZE:", len(train), len(validation), len(test))

        # CREATE MODEL
        if type(model_name) is list:
            model = Sequential(model_name)
            optimizer = Adam(learning_rate=learning_rate)
        else:
            model = CNNModels.get_model(model_name)
            optimizer = Adam()
        model.compile(
            optimizer=optimizer, 
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # FITTING
        print("-" * 35)
        print(f">> Train fold {fold_counter+1} ...")
        print("-" * 35)

        model.fit(
            x=train,
            validation_data=validation,
            epochs=epochs,
            callbacks=[
                EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=4)
            ],
            verbose=2,
        )

        # EVALUATING
        print("-" * 35)
        print(f"<< Eval fold {fold_counter+1} ...")
        print("-" * 35)

        scores = model.evaluate(test)

        print(f"Score for fold {fold_counter+1}:")
        for i, s in enumerate(model.metrics_names):
            print(f">> {s} of {scores[i]}")

        # SAVE STUFF
        accuracies.append(scores[1])
        losses.append(scores[0])

        # THAT's ALL FOLKS
        del model
        gc.collect()

    with open(f"accuracies_{tag}.txt", "wt") as f:
        for e in accuracies:
            f.write(f"{e}\n")

    return accuracies
