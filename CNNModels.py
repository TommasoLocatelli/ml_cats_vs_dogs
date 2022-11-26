import tensorflow as tf
from tensorflow import keras

from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.utils import *

padding = "same"
activation = "relu"


def BASE():
    model = Sequential()
    model.add(Rescaling(1.0 / 255))
    model.add(
        Conv2D(
            filters=16,
            kernel_size=3,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=32,
            kernel_size=3,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=64,
            kernel_size=3,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(
        Dense(
            units=64,
            activation=activation,
        )
    )
    model.add(
        Dense(
            units=8,
            activation=activation,
        )
    )
    model.add(Dense(2))
    return model


def THICK():
    model = Sequential()
    model.add(Rescaling(1.0 / 255))
    model.add(
        Conv2D(
            filters=32,
            kernel_size=3,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=64,
            kernel_size=4,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=128,
            kernel_size=5,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(
        Dense(
            units=128,
            activation=activation,
        )
    )
    model.add(
        Dense(
            units=16,
            activation=activation,
        )
    )
    model.add(Dense(2))
    return model


def LONG():
    model = Sequential()
    model.add(Rescaling(1.0 / 255))
    model.add(
        Conv2D(
            filters=16,
            kernel_size=7,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=24,
            kernel_size=5,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=32,
            kernel_size=4,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=48,
            kernel_size=3,
            padding=padding,
            activation=activation,
        )
    )
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(
        Dense(
            units=64,
            activation=activation,
        )
    )
    model.add(
        Dense(
            units=8,
            activation=activation,
        )
    )
    model.add(Dense(2))
    return model


def get_model(model_name: str):
    model_name = model_name.upper()
    if model_name == "BASE":
        return BASE()
    if model_name == "THICK":
        return THICK()
    if model_name == "LONG":
        return LONG()
    if model_name == "_":
        print("!!! NO MODEL FOUND, USING DEFAULT BASE !!!")
        return None
