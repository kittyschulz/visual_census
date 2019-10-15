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

# Call devkit() to get variables for writing records
_, cars_train_labels, __, ___ = format_data.devkit()

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def functionalize(path):
  record_file = 'images.tfrecords'
  with tf.io.TFRecordWriter(record_file) as writer:

    for filename, label in cars_train_labels.items():

      image_string = open(path+filename, 'rb').read()
      tf_example = image_example(image_string, label)
      writer.write(tf_example.SerializeToString())

def read_record(path):
  functionalize(path)
  raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

  # Create a dictionary describing the features.
  image_feature_description = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'label': tf.io.FixedLenFeature([], tf.int64),
      'image_raw': tf.io.FixedLenFeature([], tf.string),
  }

  def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

  parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
  return parsed_image_dataset

