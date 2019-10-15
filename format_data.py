from __future__ import absolute_import, division, print_function, unicode_literals

# Import system helpers
import os
import pathlib
from os import listdir
from os.path import isfile, join

# Import TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import helper libraries
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image

def label_file_names(copy_path): 
    """
    label_file_names() renames the image files
    with a new naming convention that includes the 
    image's label.
    """
    i = 0
      
    for filename in os.listdir(copy_path):
        dst = str(i) + '_' + str(test[filename]) + ".jpg"
        src = copy_path + filename 
        dst = copy_path + dst 
        os.rename(src, dst) 
        i += 1

