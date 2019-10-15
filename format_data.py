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
        dst = str(i) + '_' + str(cars_train_labels[filename]) + ".jpg"
        src = copy_path + filename 
        dst = copy_path + dst 
        os.rename(src, dst) 
        i += 1

def get_meta_data(sc_devkit = '/Users/katerina/Workspace/visual_census/data/devkit'):
    
    # Convert .mat to Python dict with Scipy.io
    cars_meta = scipy.io.loadmat(os.path.join(sc_devkit, 'cars_meta.mat'))['class_names']
    cars_train_annos = scipy.io.loadmat(os.path.join(sc_devkit, 'cars_train_annos.mat'))['annotations']
    cars_test_annos = scipy.io.loadmat(os.path.join(sc_devkit, 'cars_test_annos.mat'))['annotations']
    
    # Extract Car Metadata from dictionary to an array
    car_makes = []
    for idx, vehicle in enumerate(cars_meta[0]):
        car_makes.append([vehicle[0],vehicle[0].split(' ')[0], vehicle[0].split(' ')[-1]])
    car_makes = pd.DataFrame(car_makes, columns=['full_label', 'mnfr', 'year'])

    # Get array of only numerical labels
    car_make_nums = np.array(car_makes.index)

    # Put training data annotations into pd DataFrame and training image labels in np arr
    cars_train_labels = {}
    cars_train_annotations = []
    for idx, anno in enumerate(cars_train_annos[0]):
        cars_train_labels[anno[5][0]] = anno[4][0][0]
        cars_train_annotations.append([anno[0][0][0], anno[1][0][0], anno[2][0][0], anno[3][0][0], anno[4][0][0], anno[5][0]])
    cars_train_annotations = pd.DataFrame(cars_train_annotations, columns=['bb0', 'bb1', 'bb2', 'bb3', 'label', 'img_name'])

    # Put testing data annotations into pd DataFrame
    cars_test_annotations = []
    for idx, anno in enumerate(cars_test_annos[0]):
        cars_test_labels.append(anno[4][0][0])
        cars_test_annotations.append([anno[0][0][0], anno[1][0][0], anno[2][0][0], anno[3][0][0], anno[4][0]])
    cars_test_annotations = pd.DataFrame(cars_test_annotations, columns=['bb0', 'bb1', 'bb2', 'bb3', 'img_name'])

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(tf.strings.split(file_path, '_')[-1], '.')[0]
  # The last is the image name
  return parts==np.array([str(idx) for idx in car_makes.index])

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label