# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import os
import zipfile
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
import pickle

# For drawing onto the image.
import numpy as np

# Other packages
from tqdm import tqdm


def load_img(path):
    """
    loads and decodes image using tf
    
    Args:
        path (str): path to image (*.jpeg) file
    
    Returns:
        tensor of decoded *.jpeg
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, path, confidence=0.25):
    """ 
    run_detector calls load_img() to decode a pillow image, runs object 
    detector on the image, and builds a dictionary with the results. 
    Only car-type objects detected that fall above the input confidence 
    level are saved to the output dictionary.

    Args: 
        detector (str): the chosen detector (assumed to be from tfhub module)
        path (str): path to image
        confidence (flt): a number between 0 and 1

    Returns:
        detected_cars (dict): dictionary of image names with their assigned class, 
        bbox coordinates, and score.
    """
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[
        tf.newaxis, ...]
    result = detector(converted_img)
    result = {key: value.numpy() for key, value in result.items()}

    detected_cars = []
    for i, _ in enumerate(result):
        if (result["detection_scores"][i] > confidence) & (result["detection_class_entities"][i] == str.encode('Car')):
            object_dict = {'class': result["detection_class_entities"][i],
                           'bbox': result["detection_boxes"][i],
                           'score': result["detection_scores"][i]}
            detected_cars.append(object_dict)

    return detected_cars


def main():
    # @param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures['default']

    image_dict = {}
    ucf_data_path = '../ucf_data/' 
    image_list = [f for f in os.listdir(ucf_data_path) if f[-3:] == 'jpg']
    for image in tqdm(image_list):
        image_path = os.path.join(ucf_data_path, image)
        image_dict[image] = run_detector(detector, image_path)

    with open('ucf_objects_detected.pickle', 'wb') as handle:
        pickle.dump(image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
