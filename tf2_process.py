from __future__ import absolute_import, division, print_function, unicode_literals

# Import system helpers
import os
import pathlib
from os import listdir
from os.path import isfile, join

# Import TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Import helper libraries
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image

# Import pipeline
import format_data
import write_tf2record

def process_data(path):
    # Call devkit() to get variables for processing pipeline 
    car_makes, cars_train_labels, cars_train_annotations, cars_test_annotations = format_data.devkit()

    # Get parsed image dataset from write_tf2record pipeline
    parsed_image_dataset = write_tf2record.read_record(path)

    # Extract labels and images
    labels = np.array([image_features['label'].numpy() for image_features in parsed_image_dataset])
    images = np.array([image_features['image_raw'].numpy() for image_features in parsed_image_dataset])
    
    # Resize images
    images = [format_data.decode_img(img) for img in images[:1000]]

