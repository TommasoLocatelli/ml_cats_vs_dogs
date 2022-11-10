# %% [markdown]
# # CNN 😎

# %% [markdown]
# #### Libraries

# %%
import utilities as ff
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path, PurePath
import math


# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow import config as tfconfig

from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *


# %%
from tensorflow.python.client import device_lib
try:
    print(tfconfig.list_physical_devices())
    print(device_lib.list_local_devices())
except:
    pass

# %%
no_classes = 2
seed = 42

# %%
ff.delete_from_list()

# %% [markdown]
# #### HyperParameter

# %%
hyper_point_batch_size = [32, 128, 512, ]
hyper_point_image_squared_size = [32, 64, 128, 256, ]
hyper_point_color_mode = {1:"grayscale", 3:"rgb"}
hyper_WIDTH = [8, 32, 128]
hyper_NLAYER = [2,3,4]

MIN_WIDTH = 8

for batch_size in hyper_point_batch_size:
    for image_squared_size in hyper_point_image_squared_size:
        for color_mode_size in hyper_point_color_mode:
            for WIDTH in hyper_WIDTH:
                for NLAYER in hyper_NLAYER:

                    # %%
                    # # fixed hyper
                    # batch_size = hyper_point_batch_size[1]
                    # image_squared_size = hyper_point_image_squared_size[2]
                    # color_mode_size = 1
                    # WIDTH, NLAYER = hyper_WIDTH[1], hyper_NLAYER[2]


                    color_mode = hyper_point_color_mode[color_mode_size]
                    WHIDTH_MARGIN = math.log2(WIDTH/MIN_WIDTH)/(NLAYER-1)

                    # %%
                    TAG = f"{image_squared_size}_img_{color_mode}_{batch_size}_batch_{WIDTH}_filters_{NLAYER}_layers"
                    model_directory = Path(TAG)
                    model_directory.mkdir(parents=True, exist_ok=True)

                    # %% [markdown]
                    # #### Dataset

                    # %%
                    train, test = ff.ready_to_be_used_dataset(
                        batch_size=batch_size,
                        image_squared_size=image_squared_size,
                        color_mode=color_mode,
                    )

                    true_train_sike = int(len(train)*0.8)
                    train,valid=train.take(true_train_sike), train.skip(true_train_sike)

                    # %%
                    # image_batch, labels_batch = next(iter(train))
                    # print(image_batch.shape, labels_batch.shape)


                    # %% [markdown]
                    # #### RGB Model definition
                    # 
                    # https://www.tensorflow.org/tutorials/images/cnn

                    # %%
                    model = Sequential()
                    # model.delete-everything.forever(2022)

                    # %%
                    for i in range(NLAYER-1):
                        NFILTERS = WIDTH - int(i*2**WHIDTH_MARGIN)
                        model.add(Conv2D(NFILTERS, (3, 3), activation="relu", input_shape=(image_squared_size, image_squared_size, color_mode_size)))
                        model.add(MaxPooling2D((2, 2)))

                    model.add(Conv2D(8, (3, 3), activation="relu", input_shape=(image_squared_size, image_squared_size, color_mode_size)))
                    model.add(MaxPooling2D((2, 2)))
                    model.add(Flatten())


                    # %%
                    for i in range(NLAYER):
                        NFILTERS = WIDTH - int(i*2**WHIDTH_MARGIN)
                        model.add(Dense(NFILTERS, activation="relu"))

                    model.add(Dense(2))

                    # %%
                    # Open the file
                    with open(PurePath(TAG, 'model_summary.txt'), 'wt') as fh:
                        # Pass the file handle in as a lambda function to make it callable
                        model.summary(print_fn=lambda x: fh.write(x + '\n'))

                    # %%
                    model.compile(
                        optimizer="adam",
                        loss=SparseCategoricalCrossentropy(from_logits=True),
                        metrics=["accuracy"],
                    )

                    # %%
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


                    # %% [markdown]
                    # #### Training

                    # %%
                    history = model.fit(train, epochs=25, validation_data=valid)

                    # %% [markdown]
                    # #### Performance evaluation

                    # %%
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
                        plt.savefig(PurePath(TAG, 'loss_accuracy.png'))

                    performance_plot(history)

                    # %% [markdown]
                    # ### Risk Estimation

                    # %%
                    accuracies = ff.five_fold_cross_validation(model, test, no_epochs=25)
                    with open(PurePath(TAG, 'accuracies.txt'), 'wt') as f:
                        for e in accuracies:
                            f.writeline(e)
