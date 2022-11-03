# %%
import numpy as np
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *

# %%



batch_size=1
img_width, img_height=100,100
color_mode='grayscale'
no_epochs=1
loss_function = 'sparse_categorical_crossentropy'
no_classes = 2
optimizer = Adam()
verbosity = 1
num_folds = 5
seed=123
data_dir = pathlib.Path("downloads\CatsDogs")

# %%


ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    color_mode=color_mode,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
# %%

inputs = np.array([i[0] for _,(i,l) in enumerate(ds)])
targets = np.array([l[0] for _,(i,l) in enumerate(ds)])

# %%

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []   

# Define the K-fold Cross Validator
kfold = KFold(n_splits=2, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    model = Sequential(
        [
            
            Flatten(input_shape=(100, 100, 1)),
            Dense(100, activation="relu"),
            Dense(2, activation="softmax"),
        ]
    )
    # Compile the model
    model.compile(loss=loss_function,
                    optimizer=optimizer,
                    metrics=['accuracy'])
    # Fit data to model
    history = model.fit(inputs[train], targets[train],
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity)

# %%
