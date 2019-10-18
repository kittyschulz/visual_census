# Visual Census Using Car-Type Image Classifier on Street-Level Imagery

## 1. Fine-Grained Car Classification

The first step in performing this Visual Census was to fine tune a pre-built image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). the Keras Sequential model API was used. To train the classifier on the Stanford dataset, each image was cropped based on the bounding boxes provided in the training set and resized to 224 by 224 pixels. We used vehicle car type labels to train the model. 

Our ResNet50 model obtained a validation accuracy of 66 percent. The model contains a total of xx layers.

[include model summary]

### Features of the Stanford Cars Dataset:

- 8,144 labeled training images in a 16,185 image dataset.

- 196 unique car labels (make-model-year).

- The car models contained in the dataset all originate from the North American market.

- Average of 41 distinct images per make-model label in training set.

- Dataset includes multiple points of view for each car.

![](/Users/katerina/Workspace/visual_census/presentation_plots/stanford_sample.png)

## 2. Object Detection on Street-Level Scenes

The car-type objects were counted and cropped from selected scenes of the [UCF Google Streetview data](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset) through the use of a [TF-Hub module](https://www.tensorflow.org/hub/overview) trained to perform object detection. For each car-type object detected, the object was cropped from the scene if and only if it had a detector confidence of 20-percent or greater.

![](/Users/katerina/Workspace/visual_census/presentation_plots/obj_detector.png)

Other object types, including pedestrians, cyclists, and buses were ignored. The bounding boxes obtained from the object detector were used to "crop" each street-level image to isolate each car. We then ran our image classifier on each of the isolated car images. 

The object detector was run on a subset of approximately 25% of the scenes (2,500 in total) to eliminate the redundancy of consectutive scenes. The scenes had a median count of 12 vehicles and a maximum of 39 vehicles per scene. Five (5) percent of the scenes contained less than five vehicles. 

![](/Users/katerina/Workspace/visual_census/presentation_plots/number_of_cars.png)

### Features of UCF Google StreetView Dataset:

- 62,058 Google Street View images covering areas of Pittsburgh, Orlando, partially Manhattan for 10,343 individual scenes.

- Five views per scene: four side views, one sky view, one repeated side view with annotations.

- Each image includes accurate GPS coordinates and compass direction.

![](/Users/katerina/Workspace/visual_census/presentation_plots/street_view_sample.png)

## 3. Map Predictions

Cars were classified based on both their Value (Economy, Standard, Luxury. Exotic) and Type (SUV, Sedan, Coupe, etc.). The goal of this visual census was to estimate the relative socio-economic status of each neighborhood traversed in our street-level data using these classifications. 

Approximately 1,500 unqiue cars were classified out of the 2,500 scenes examined.

## Results

- Maximum validation accuracy on Value classification (68%) 
- Achieved 66% validation accuracy on Type classification
- Results and visual analysis show lack of diversity in car Types and Value

### Limitations:

- Did not account for yellow cabs, police cars, or commercial vehicles

- Parked cars and moving cars are given the same weight in our prediction

- Lack of diversity in car types of images inspected

- Relatively low resolution of object instances makes a fine-grained classification difficult

- Relatively low accuracy of fine-grained image classifier

