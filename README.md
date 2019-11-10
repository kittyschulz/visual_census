# Visual Census Using Car-Type Image Classifier on Street-Level Imagery

The goal of this work is to use a fine-grained car model image classifier and object detector on geo-localized images to determine if they can provide meaningful census signals.

We build a fine-grained image classifier by fine-tuning a ResNet152 image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

## 1. Fine-Grained Car Classification

The first step in performing this Visual Census was to fine tune a pre-built image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). 

### The Stanford Cars Dataset

![stanford](/sample_images/stanford_sample.png)

The dataset is comprised of a total of 16,185 images with 196 unique labels of car make-model-year. The data is split into a 8,144-image training set and 8,041-image testing set. There is an average of 41 distinct images per label in the training set, which include multiple points of view for each car label.

The car models contained in the dataset were all avliable on the North American market. The models ranged from cars like the [Geo Metro](https://en.wikipedia.org/wiki/Geo_Metro) to the [Bugatti Veyron](https://en.wikipedia.org/wiki/Bugatti_Veyron). Model years ranged from 1991 to 2012. 

For our purposes, the Stanford Cars Dataset contains a slightly disproportionate amount of exotic or ultra-luxury cars (cars costing more than $200,000USD). Of the 8,144 images in the training set, 1,072 can be classified as 'exotic' or 'ultra-luxury' cars--about 13 percent. 

### ResNet152 Image Classifier

A ResNet152 model pretrained on ImageNet was fine tuned on the Stanford Cars Dataset. 



### Other Models

the Keras Sequential model API was used. To train the classifier on the Stanford dataset, each image was cropped based on the bounding boxes provided in the training set and resized to 224 by 224 pixels. We used vehicle car type labels to train the model. 

Our ResNet50 model obtained a validation accuracy of 66 percent for car type labels.

The Keras Sequential API was also used as a faster alternative to the RestNet model. It was trained on the car Value labels and ultimately obtained a validation accuracy of 68 percent.

- Maximum validation accuracy on Value classification (68%) 
- Achieved 66% validation accuracy on Type classification

## 2. Object Detection on Street-Level Scenes

The car-type objects were counted and cropped from scenes of the [UCF Google Streetview data](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset) through the use of a [TF-Hub module](https://www.tensorflow.org/hub/overview) trained to perform object detection. For each car-type object detected, the object was cropped from the scene if and only if it had a detector confidence of 25-percent or greater.

![obj detect](/sample_images/obj_detector.png)

Other object types, including pedestrians, cyclists, and buses were ignored. The bounding boxes obtained from the object detector were used to "crop" each street-level image to isolate each car. We then ran our image classifier on each of the isolated car-type objects. 

The scenes had a median count of 12 vehicles and a maximum of 39 vehicles per scene. Five (5) percent of the scenes contained less than five vehicles. 

![num cars](/sample_images/number_of_cars.png)

### The UCF Google StreetView Dataset:

![streetview](/sample_images/street_vew_sample.png)

The UCF Google StreetView Dataset is comprised of 62,058 Google Street View images covering areas of Pittsburgh, Orlando, partially Manhattan for 10,343 individual scenes.

- Five views per scene: four side views, one sky view, one repeated side view with annotations.

- Each image includes accurate GPS coordinates and compass direction.


## 3. Map Predictions

Cars were classified based on both their Value (Economy, Standard, Luxury. Exotic) and Type (SUV, Sedan, Coupe, etc.). The goal of this visual census was to estimate the relative socio-economic status of each neighborhood traversed in our street-level data using these classifications. 

Approximately 1,500 unqiue cars were classified out of the 2,500 scenes examined.

## Results


### Limitations


- Did not account for yellow cabs, police cars, or commercial vehicles

- Parked cars and moving cars are given the same weight in our prediction

### Future Work

We hope that this work will be expanded in the future to additional neighborhoods in lower Manhattan. We hope to accomplish this by use of the Google StreetLearn dataset

