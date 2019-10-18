from __future__ import absolute_import, division, print_function, unicode_literals

# Import system helpers
import os
import pathlib
from os import listdir
from os.path import isfile, join

# Import TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import helper libraries
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image

IMG_HEIGHT, IMG_WIDTH = 32, 32

def read_record(tfrecord='images.tfrecords'):
  raw_image_dataset = tf.data.TFRecordDataset(tfrecord)

  # Create a dictionary describing the features.
  image_feature_description = {
      'image_raw': tf.io.FixedLenFeature([], tf.string),
      'mnfr_label': tf.io.FixedLenFeature([], tf.string),
      'year_label': tf.io.FixedLenFeature([], tf.string),
      'full_label': tf.io.FixedLenFeature([], tf.string),
      'bbox': tf.io.FixedLenFeature([4,], tf.float32),
  }

  def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

  parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
  return parsed_image_dataset

def label_sets(sc_devkit):
    cars_meta = scipy.io.loadmat(os.path.join(sc_devkit, 'cars_meta.mat'))['class_names']
    car_makes = []
    for vehicle in cars_meta[0]:
        car_makes.append([vehicle[0],vehicle[0].split(' ')[0], vehicle[0].split(' ')[-1]])
    car_makes = pd.DataFrame(car_makes, columns=['full_label', 'mnfr', 'year'])

    return car_makes

def create_data(sc_devkit, ds, n, split=0.8):
    # Get percent of data to split on
    n_split = int(n*split)

    # Get pd DataFrame of car label data
    car_makes = label_sets(sc_devkit)

    # Get unique values for mnfr, year, and full label names
    car_mnfr = list(set(car_makes['mnfr']))
    car_year = list(set(car_makes['year']))
    car_full_name = list(set(car_makes['full_label']))

    # Create empty lists to write labels and image data to 
    images = []
    mnfr_labels = []
    year_labels = []
    full_labels = []

    # Iterate over tfRecord to get data
    for features in ds.take(n):
        image = tf.image.decode_image(features['image_raw'], channels=3)
        H, W, _ = image.shape
        bbox = features['bbox']
        cropped = tf.image.crop_and_resize(
            [image],
            [[bbox[0]/H, bbox[1]/W, bbox[2]/H, bbox[3]/W]],
            [0,],
            [224, 224])
        cropped = tf.image.resize(cropped, [IMG_HEIGHT, IMG_WIDTH])
        images.append(cropped.numpy())


        mnfr = features['mnfr_label'].numpy()
        mnfr_labels.append(mnfr.decode())

        year = features['year_label'].numpy()
        year_labels.append(year.decode())

        full = features['full_label'].numpy()
        full_labels.append(full.decode())

    images = np.squeeze(images)
    # Convert string labels to numerical labels
    mnfr_labels = [car_mnfr.index(label) for label in mnfr_labels]
    year_labels = [car_year.index(label) for label in year_labels]
    full_labels = [car_full_name.index(label) for label in full_labels]

    # Split test and train sets for images and labels
    train_images, test_images = np.array(images[:n_split]), np.array(images[n_split:])
    train_labels, test_labels = np.array(mnfr_labels[:n_split]), np.array(mnfr_labels[n_split:])

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, test_images, train_labels, test_labels

def train_model(train_images, test_images, train_labels, test_labels):
    # Build model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(196, activation='softmax'))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))
    
ds = read_record()
train_images, test_images, train_labels, test_labels = create_data(sc_devkit='/Users/katerina/Workspace/visual_census/data/devkit', ds=ds, n=8000, split=0.8)
train_model(train_images, test_images, train_labels, test_labels)