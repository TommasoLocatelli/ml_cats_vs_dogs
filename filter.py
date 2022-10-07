# %%
import gc
import os
import warnings
import imghdr
import cv2
import glob

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from PIL import Image

# %%
import tensorflow as tf
from tensorflow import keras

from keras.layers import *
from keras.models import *
from keras.losses import *

from keras.utils import image_dataset_from_directory, load_img, img_to_array

print(tf.config.list_physical_devices('GPU'))

# %%
CAT, DOG = 'Cats', 'Dogs'
uri = 'downloads/{}/{}.jpg'

# %% [markdown]
# # baumiao

# %%
input_shape = (256, 256, 3) # (heigt, width, D)
image_size = input_shape[:2]
batch_size = 32
N_channels = input_shape[2]
epochs = 20

# %%
exp_ext = set()
ds = np.array([])

# %% [markdown]
# ## import dataset

# %%
# method 1
warnings.filterwarnings("error")
bad_paths = []

for e in [CAT,DOG]:
    for pth in Path(f'downloads/{e}').rglob("*"):
        
        ext = imghdr.what(pth)
        if ext is None:
            print(pth,'removed')
            os.remove(pth)
        else:
            exp_ext.add(ext)
            try:
                with Image.open(pth) as img:
                    pxl = np.array(img)
                    if pxl.ndim < 3 or pxl.shape[-1] < 3:
                    # if pxl.ndim != 2:
                        bad_paths.append(pth)
                    else:
                        pass
            except Exception as e:
                bad_paths.append(pth)
                print(pth,e)

warnings.filterwarnings("default")

# %%
# method 2
bad_paths = []
shapeset = set()

img_paths = glob.glob(os.path.join('downloads','*/*.*')) # assuming you point to the directory containing the label folders.
for image_path in img_paths:

    try:
        img = load_img(image_path, target_size=image_size)
        img = img_to_array(img)
        shapeset.add(img.shape)
        # img_bytes = tf.io.read_file(image_path)
        # decoded_img = tf.decode_image(img_bytes)
    except Exception as inst:
        print('trouble at', image_path, ':', inst)
        bad_paths.append(image_path)

# %%
# method 3
bad_paths = []

for folder_name in ("Cats", "Dogs"):
    folder_path = os.path.join("downloads", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            print(f"Found bad path {fpath}")
            bad_paths.append(fpath)

# %%
len(bad_paths)

# %%
for pth in bad_paths:
    try:
        os.remove(pth)
        print(pth,'removed')
    except Exception as e:
        print('FATAL ERROR @', pth, ':', e)