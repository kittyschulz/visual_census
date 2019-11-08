# For loading the images and annotations
import json
import os
import pickle

# For preprocessing each image
import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io

# For running our classifier
from utils import load_model

if __name__ == '__main__':
    img_width, img_height = 224, 224
    model = load_model()
    model.load_weights('models/model.96-0.89.hdf5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    test_path = 'ucf_data/' #'/Users/katerina/Workspace/visual_census/ucf_data/'

    # use either resnetv2 or mobilenet; resnet pickle should contain better results
    with open('object_detector/ucf_objects_detected_inception_resnet_v2.pickle', 'rb') as f: #/Users/katerina/Workspace/visual_census/
        ucf_objects_resnet = pickle.load(f)

    # get the count of object instances in each scene (used to test heatmap)
    object_counts = {}
    # create empty list to write our results to
    results = []

    for key, value in ucf_objects_resnet.items():
        # read in whole street view images
        object_counts[key] = len(value)

        # we can ignore scenes with no other car-type object instances
        if len(value) > 0:
            src_path = os.path.join(test_path, key)
            print('Start processing image: {}'.format(key))
            src_image = cv.imread(src_path)
            #image_pil = Image.fromarray(np.uint8(src_image)).convert("RGB")
            height, width = src_image.shape[:2]

            # iterate over object instances in scene
            for instance in range(len(value)):
                # crop object instance with bbox
                bbox = ucf_objects_resnet[key][instance]['bbox']
                ymin, xmin, ymax, xmax = tuple(bbox)
                ymax = height*(ymax - ymin)
                xmax = width*(xmax - xmin)
                ymin = ymin*height
                xmin = xmin*width

                margin = 16
                x1 = int(max(0, xmin - margin))
                y1 = int(max(0, ymin - margin))
                x2 = int(min(xmax + margin, width) + x1)
                y2 = int(min(ymax + margin, height) + y1)
                crop_image = src_image[y1:y2, x1:x2]

                # preprocess cropped image for classifier
                dst_img = cv.resize(src=crop_image, dsize=(img_width, img_height), interpolation=cv.INTER_CUBIC)
                rgb_img = cv.cvtColor(dst_img, cv.COLOR_BGR2RGB)
                rgb_img = np.expand_dims(rgb_img, 0)

                # make predictions
                preds = model.predict(rgb_img)
                prob = np.max(preds)
                class_id = np.argmax(preds)
                text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
                print(text)
                results.append({'scene': key, 'label': class_names[class_id][0][0], 'prob': '{:.4}'.format(prob)})

    # write results to json file
    with open('results_resnet_v2.json', 'w') as file:
        json.dump(results, file, indent=4)

    # clear session to avoid clutter from old models / layers
    K.clear_session()
