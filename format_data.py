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
        dst = str(cars_train_labels[filename]) + '_' + filename
        src = copy_path + filename 
        dst = copy_path + dst 
        os.rename(src, dst) 
        i += 1

def devkit(devkit_path='/Users/katerina/Workspace/visual_census/data/devkit'):
    sc_devkit = devkit_path
        
    # Convert .mat to Python dict with Scipy.io
    cars_meta = scipy.io.loadmat(os.path.join(sc_devkit, 'cars_meta.mat'))['class_names']
    cars_train_annos = scipy.io.loadmat(os.path.join(sc_devkit, 'cars_train_annos.mat'))['annotations']
    cars_test_annos = scipy.io.loadmat(os.path.join(sc_devkit, 'cars_test_annos.mat'))['annotations']

    # Extract Car Metadata from dictionary to an array
    car_makes = []
    for vehicle in cars_meta[0]:
        car_makes.append([vehicle[0],vehicle[0].split(' ')[0], vehicle[0].split(' ')[-1]])
    car_makes = pd.DataFrame(car_makes, columns=['full_label', 'mnfr', 'year'])

    # Put training data annotations into pd DataFrame and training image labels in np arr
    # cars_train_labels contains a dictionary with keys of the image name and values of the numerical label
    cars_train_labels = {}
    # cars_train_annotations is a pd DataFrame with columns bb(x1), bb(x2), bb(y1), bb(y2), numerical label and image name
    cars_train_annotations = []
    for anno in cars_train_annos[0]:
        cars_train_labels[anno[5][0]] = anno[4][0][0]
        cars_train_annotations.append([anno[0][0][0], anno[1][0][0], anno[2][0][0], anno[3][0][0], anno[4][0][0], anno[5][0]])
    cars_train_annotations = pd.DataFrame(cars_train_annotations, columns=['bb_x1', 'bb_x2', 'bb_y1', 'bb_y2', 'label', 'img_name'])

    # Put testing data annotations into pd DataFrame
    # cars_test_annotations is a pd DataFrame with columns bb(x1), bb(x2), bb(y1), bb(y2), and image name
    cars_test_annotations = []
    for anno in cars_test_annos[0]:
        cars_test_annotations.append([anno[0][0][0], anno[1][0][0], anno[2][0][0], anno[3][0][0], anno[4][0]])
    cars_test_annotations = pd.DataFrame(cars_test_annotations, columns=['bb_x1', 'bb_x2', 'bb_y1', 'bb_y2', 'img_name'])

    return car_makes, cars_train_labels, cars_train_annotations, cars_test_annotations

# Call devkit() to get variables for processing pipeline 
car_makes, cars_train_labels, cars_train_annotations, cars_test_annotations = devkit()

# Define BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, STEPS_PER_EPOCH and AUTOTUNE for prep for training
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
AUTOTUNE = tf.data.experimental.AUTOTUNE
#STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(tf.strings.split(file_path, '_')[-1], '.')[0]
    # The last is the image name
    return parts==np.array([str(idx) for idx in car_makes.index])

def decode_img(img, IMG_WIDTH=224, IMG_HEIGHT=224):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=32):
    """
    prepare_for_training() shuffles a given ds and 
    feteches and returns a batch of size BATCH_SIZE.
    """
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # 'prefetch' lets the dataset fetch batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def get_batch(file_path):
    #'/Users/katerina/Workspace/visual_census/data/training_data/cars_train copy/*'
    list_ds = tf.data.Dataset.list_files(str(file_path))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    prep_ds = prepare_for_training(labeled_ds)
    image_batch, label_batch = next(iter(prep_ds))

    return image_batch, label_batch

def crop_img(img):
    """
    crop_img() takes the bounding boxes of an object 
    in a given image file and crops the image.
    """
    pass

