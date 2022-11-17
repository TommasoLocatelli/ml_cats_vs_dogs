# %%
import os
from pathlib import Path, PurePath
from tqdm import tqdm

# %%
import tensorflow as tf

from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *

from tensorflow.errors import InvalidArgumentError

# %%
root = PurePath("downloads/CatsDogs")
temp_root = PurePath(root, "tmp")
Path(temp_root, "Cats").mkdir(parents=True, exist_ok=True)
Path(temp_root, "Dogs").mkdir(parents=True, exist_ok=True)

# %%
size = 128
batch_size = 1

model = Sequential(
    [
        Flatten(input_shape=(size, size, 1)),
        Dense(1)
    ]
)

model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# %%
for razza in ["Cats", "Dogs"]:
    for i in tqdm(range(12500)):

        filename = f"{i}.jpg"
        orig_path = PurePath(root, razza, filename)
        temp_path = PurePath(temp_root, razza, filename)

        try:
            os.rename(orig_path, temp_path)
        except:
            print(orig_path, "not found (already deleted?)")
            continue

        train_ds = image_dataset_from_directory(
            temp_root,
            color_mode="grayscale",
            image_size=(size, size),
            batch_size=batch_size,
        )

        try:
            model.fit(train_ds, epochs=1, verbose=0)
        except InvalidArgumentError:
            with open("files.to.delete.txt", "a") as f:
                f.write(orig_path + "\n")
        finally:
            os.rename(temp_path, orig_path)
