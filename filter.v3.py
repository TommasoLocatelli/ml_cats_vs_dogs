
# %%
from utilities import *
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *

from tensorflow.errors import InvalidArgumentError

files_to_delete = list()

# %%
import os
from tqdm import tqdm
pth = "downloads/CatsDogs/{}"
data_dir = pathlib.Path(pth.format("tmp/"))


# %%
for razza in ['Cats', 'Dogs', ]:
    for i in tqdm(range(12500)):
        
        orig_path = pth.format("")+ razza+f"/{i}.jpg"
        temp_path = pth.format("tmp/")+razza+f"/{i}.jpg"
        try:
            os.rename(orig_path, temp_path)
        except:
            print(orig_path, "already deleted")
            continue

        image_squared_size=256
        batch_size = 1
        img_height = image_squared_size
        img_width = image_squared_size

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            # validation_split=0.2,
            color_mode='grayscale',
            # subset="training",
            # seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        );

        # normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
        # normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))

        train_ds

        
        model = Sequential(
            [
                Flatten(input_shape=(image_squared_size, image_squared_size, 1)),
                Dense(10, activation="relu"),
                Dense(2, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        
        try:
            model.fit(train_ds, epochs=1, verbose=0)
        except InvalidArgumentError:
            # print('removing', temp_path)
            # os.remove(temp_path)
            files_to_delete.append(orig_path)
        finally:
            os.rename(temp_path, orig_path)
        
with open('files.to.delete.txt', 'wt') as f:
    f.writelines([l+'\n' for l in files_to_delete])
        