# %%
from utilities import *

import tensorflow as tf
from tensorflow import keras

from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *

from tensorflow.errors import InvalidArgumentError

# %%
import os
from tqdm import tqdm

pth = "downloads/CatsDogs/{}"
data_dir = pathlib.Path(pth.format("tmp/"))

# %%
size = 128
batch_size = 1

model = Sequential(
    [
        Flatten(input_shape=(size, size, 1)),
    ]
)

model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)


# %%
for razza in [
    "Cats",
    "Dogs",
]:
    for i in tqdm(range(12500)):

        orig_path = pth.format("") + razza + f"/{i}.jpg"
        temp_path = pth.format("tmp/") + razza + f"/{i}.jpg"
        try:
            os.rename(orig_path, temp_path)
        except:
            print(orig_path, "already deleted")
            continue

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            # validation_split=0.2,
            color_mode="grayscale",
            # subset="training",
            # seed=seed,
            image_size=(size, size),
            batch_size=batch_size,
        )

        # normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        # normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))

        try:
            model.fit(train_ds, epochs=1, verbose=0)
        except InvalidArgumentError:
            # print('removing', temp_path)
            # os.remove(temp_path)
            with open("files.to.delete.txt", "a") as f:
                f.write(orig_path + "\n")
        finally:
            os.rename(temp_path, orig_path)
