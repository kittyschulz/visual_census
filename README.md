# Visual Census Using Car-Type Image Classifier on Street-Level Imagery

The goal of this work is to use a fine-grained car model image classifier and object detector on geo-localized images to determine if they can provide meaningful census signals.

We build a fine-grained image classifier by fine-tuning a ResNet152 image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

## 1. Fine-Grained Car Classification

The first step in performing this Visual Census was to fine tune a pre-built image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). 

### The Stanford Cars Dataset

![stanford](/img/stanford_sample.png)

The dataset is comprised of a total of 16,185 images with 196 unique labels of car make-model-year. The data is split into a 8,144-image training set and 8,041-image testing set. There is an average of 41 distinct images per label in the training set, which include multiple points of view for each car label.

The car models contained in the dataset were all avliable on the North American market. The models ranged from cars like the [Geo Metro](https://en.wikipedia.org/wiki/Geo_Metro) to the [Bugatti Veyron](https://en.wikipedia.org/wiki/Bugatti_Veyron). Model years ranged from 1991 to 2012. 

For our purposes, the Stanford Cars Dataset contains a slightly disproportionate amount of exotic or ultra-luxury cars (cars costing more than $200,000USD). Of the 8,144 images in the training set, 1,072 can be classified as 'exotic' or 'ultra-luxury' cars--about 13 percent. 

### Initial Models

Initial benchmark models were built using a ResNet50 model pre-trained on ImageNet and the Keras Sequential model API. The Keras Sequential API was used as a faster alternative to the RestNet model.

Using the original 196 labels of the Stanford Cars Dataset, we obtained a validation accuracy on the order of 20 percent with the ResNet50 model. 
The benchmark models were used to classify 
- Maximum validation accuracy on Value classification (68%) 
- Achieved 66% validation accuracy on Type classification

To train the classifier on the Stanford dataset, each image was cropped based on the bounding boxes provided in the training set and resized to 224 by 224 pixels. We used vehicle car type labels to train the model. 

Our ResNet50 model obtained a validation accuracy of 66 percent for car type labels.

 It was trained on the car Value labels and ultimately obtained a validation accuracy of 68 percent.


### ResNet152 Image Classifier

A ResNet152 model pretrained on ImageNet was fine tuned on the Stanford Cars Dataset.  


## 2. Object Detection on Street-Level Scenes

The car-type objects were detected in scenes of the [UCF Google Streetview data](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset) through the use of a [TF-Hub module](https://www.tensorflow.org/hub/overview) trained to perform object detection. Each car-type object detected, with a confidence of at least 25 percent, we run throug the image classifier. 

![obj detect](/img/obj_detector.png)

Other object types, including pedestrians, cyclists, and buses were ignored. The bounding boxes obtained from the object detector were used to "crop" each street-level image to isolate each car. We then ran our image classifier on each of the isolated car-type objects. 

The scenes had a median count of 12 vehicles and a maximum of 39 vehicles per scene. Five (5) percent of the scenes contained less than five vehicles. 

![num cars](/img/number_of_cars.png)

### The UCF Google StreetView Dataset:

![streetview](/img/street_vew_sample.png)

The UCF Google StreetView Dataset is comprised of 62,058 Google Street View images covering areas of Pittsburgh, Orlando, partially Manhattan for 10,343 individual scenes.

- Five views per scene: four side views, one sky view, one repeated side view with annotations.

- Each image includes accurate GPS coordinates and compass direction.


## 3. Map Predictions

Cars were classified based on both their Value (Economy, Standard, Luxury. Exotic) and Type (SUV, Sedan, Coupe, etc.). The goal of this visual census was to estimate the relative socio-economic status of each neighborhood traversed in our street-level data using these classifications. 

Approximately 1,500 unqiue cars were classified out of the 2,500 scenes examined.

# Implementation

This work can be reproduced by cloning this repository and following along below. We recommend running an instance on Google Cloud using the [Deep Learning VM](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) with at least one GPU. 

### Dependencies

To reproduce this work, the following packages must be installed:

### Helpful Tips

 - If you are running on an instance, remember to call ``` $ nohup ``` so that these scripts will run in the background, and continue running even if you lose connection to your VM.

- Monitor your GPU throughout this project by calling ``` $ watch nvidia-smi ```. As always, you can also keep track of your CPU and Memory utilization with ``` $ top ``` or ``` $ htop ```.

### Image Classifier

To get the Stanford Cars Data, call the following in your terminal:

```
$ cd ../ucf_data
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

The data will be downloaded as *.tgz files. To extract and pre-process the data, call ``` $ python3 pre-process.py ```.

With the images pre-processed, we can train the model. The trained model will be saved as an *.hdf5 file. If you don't have access to a machine with multiple GPUs or a cluster, it is recommended you let this step run overnight. 

```
$ python3 build_model.py
```

### Object Detector

To get the UCF Street View Data, call the following in your terminal. You can expect this step will take up to 6 hours to complete.

```
$ cd ../object_detector
$ python3 get_ucf_data.py
```

The data will be extracted from the zipped format and stored in the ``` ucf_data ``` folder. Therefore, the object detector can immediately be run on the images by calling ``` $ python3 ucf_detector.py ```. By default, the object detector will only store car-type objects that have a confidence above 25 percent.

The output of ``` ucf_detector.py ``` will be a pickled dictionary containing bounding box coordinates and a confidence for each of the car-type objects. This dictionary will be used to crop the images in the pre-processing step when we run our image classifier. 

### Using the Model

Now that the image classifier is trained and the car-type objects have been identified in the street view scenes, we can classify the street view vehicles and use these predictions to create some nice visualizations. 

Run the image classifier on the car-type objects detected. You can expect this step to run for 2 to 3 hours on the UCF Street View Data.

```
$ python3 image_classifier/test_streetview.py
```

The output of ``` test_streetview.py ``` will be a json file containing a predicted label for each object, the object's scene number (the first 6 characters of the image file name from which it was detected), and the probability assigned to the label. Of course, the label stored for each object is the label predicted with the highest probability. 

Using the *.json file of the prediction results, we can build our first map. Do so by calling ``` $ python3 maps/build_folium_map.py ```. We'll construct a heatmap using [Folium](https://python-visualization.github.io/folium/) of the luxury cars observed along the SteetView Car's route. 

### Limitations


- Did not account for yellow cabs, police cars, or commercial vehicles

- Parked cars and moving cars are given the same weight in our prediction

### Future Work

We hope that this work will be expanded in the future to additional neighborhoods in lower Manhattan. We hope to accomplish this by use of the Google StreetLearn dataset

