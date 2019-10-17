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

from geopy.geocoders import Nominatim

def read_ucf_data(ucf_data='/Users/katerina/Workspace/visual_census/ucf_data'):
    # Convert .mat to Python dict with Scipy.io
    gps = np.squeeze(scipy.io.loadmat(os.path.join(ucf_data, 'GPS_Long_Lat_Compass.mat'))['GPS_Compass'])

    # Get scene file names:
    image_names = []
    for i in range(1,11):
        image_names += [img for img in listdir('/Users/katerina/Workspace/visual_census/ucf_data/part{}'.format(i)) if img.split('.')[1]=='jpg']
    image_names = np.unique([img.split('_')[0] for img in image_names])

    #Make a dictionary of scene names and their locations
    img_lat_long = {}
    for i in range(image_names.shape[0]):
        img_lat_long[image_names[i]] = (gps[i][0], gps[i][1])

    return img_lat_long 

def address(lat_long):
    geolocator = Nominatim(user_agent="kitti")
    location = geolocator.reverse(str(lat_long[0][0]) +','+ str(lat_long[0][1])).address
    return location

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

def image_example(image_string, loc_string):
    feature = {
        'image_raw': _bytes_feature(image_string),
        'location': _bytes_feature(loc_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_record(img_path, ucf_data='/Users/katerina/Workspace/visual_census/ucf_data'):
    img_lat_long = read_ucf_data(ucf_data)

    detector_image_names = [img for img in listdir(img_path) if img.split('.')[1]=='jpg']

    gps = []
    for img in detector_image_names:
        gps.append( img_lat_long[img.split('_')[1]] )
    
    gps = [address(lat_long) for lat_long in gps]

    record_file = 'detected_images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:

        for loc_string, filename in zip(gps, detector_image_names):
            image_string = open(img_path+filename, 'rb').read()
            tf_example = image_example(image_string, loc_string)
            writer.write(tf_example.SerializeToString())

write_record(img_path='/Users/katerina/Workspace/visual_census/ucf_data_colab/detected_images',
              ucf_data='/Users/katerina/Workspace/visual_census/ucf_data')
