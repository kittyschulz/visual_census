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


def read_stanford_cars_data(sc_devkit='/Users/katerina/Workspace/visual_census/data/devkit'):
    # Convert .mat to Python dict with Scipy.io
    cars_meta = np.squeeze(scipy.io.loadmat(os.path.join(sc_devkit, 'cars_meta.mat'))['class_names'])
    full_label_names = [x[0] for x in cars_meta]
    mnfr_label_names = [x[0].split(' ')[0] for x in cars_meta]
    year_label_names = [x[0].split(' ')[-1] for x in cars_meta]

    annotations = np.squeeze(scipy.io.loadmat(os.path.join(sc_devkit, 'cars_train_annos.mat'))['annotations'])
    class_labels = (np.array([np.squeeze(x) for x in annotations['class']]) - 1).tolist()
    image_fnames = np.array([np.squeeze(x) for x in annotations['fname']])

    bb_y1 = np.array([np.squeeze(x) for x in annotations['bbox_y1']])
    bb_x1 = np.array([np.squeeze(x) for x in annotations['bbox_x1']])
    bb_y2 = np.array([np.squeeze(x) for x in annotations['bbox_y2']])
    bb_x2 = np.array([np.squeeze(x) for x in annotations['bbox_x2']])

    full_labels = [full_label_names[idx].encode() for idx in class_labels] 
    mnfr_labels = [mnfr_label_names[idx].encode() for idx in class_labels] 
    year_labels = [year_label_names[idx].encode() for idx in class_labels] 

    bboxes = [(y1, x1, y2, x2) for y1, x1, y2, x2 in zip(bb_y1, bb_x1, bb_y2, bb_x2)]

    return full_labels, mnfr_labels, year_labels, image_fnames, bboxes

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_features(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, mnfr_string, year_string, full_string, bbox):
    feature = {
        'image_raw': _bytes_feature(image_string),
        'mnfr_label': _bytes_feature(mnfr_string),
        'year_label': _bytes_feature(year_string),
        'full_label': _bytes_feature(full_string),
        'bbox': _float_features(bbox),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_record(img_path, sc_devkit='/Users/katerina/Workspace/visual_census/data/devkit'):
    full_labels, mnfr_labels, year_labels, image_fnames, bboxes = read_stanford_cars_data(sc_devkit)

    record_file = 'images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:

        for mnfr_string, year_string, full_string, filename, bbox in zip(mnfr_labels, year_labels, full_labels, image_fnames, bboxes):
            image_string = open(img_path+filename, 'rb').read()
            tf_example = image_example(image_string, mnfr_string, year_string, full_string, bbox)
            writer.write(tf_example.SerializeToString())

write_record(img_path='/Users/katerina/Workspace/visual_census/data/training_data/cars_train/',
              sc_devkit='/Users/katerina/Workspace/visual_census/data/devkit')
