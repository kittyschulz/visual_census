## Visual Census Using Car-Type Image Classifier on Street-Level Imagery

### 1. Fine-Grained Car Classification

The first step in performing this Visual Census was to fine tune a pre-built image classifier on the Stanford cars dataset. A ResNet50 pretrained classifier was used. To train the classifier on the Stanford dataset, each image was cropped based on the bounding boxes provided in the training set and resized to 224 by 224 pixels. We used car "class" labels to train the model. The class label (including Budget, Economy, Standard, Luxury, Exotic/Super Luxury) was created by establishing 5 distinct price ranges and assigning car manufacturers to each class using the average MSRP of their production models. 

Our ResNet50 model obtained a validation accuracy of ** percent. The model 

##### Features of the Stanford Cars Dataset:

8,144 labeled training images with a 8,144 image validation set.

196 unique car labels (make-model-year) over approximately 16,000 images. 

The car models contained in the dataset all originate from the North American market. Car years range from 1991 to 20**.

Average of 41 distinct images per label in training set.

Dataset includes multiple points of view for each car.

### 2. Object Detection on Street-Level Scenes
The car-type objects were cropped from in each scene of the UCF Google Streetview data through the use of a TF-Hub module trained to perform object detection. For each car-type object detected, the object was cropped from the scene if and only if it satisfied the conditions of (1) a detector confidence of 50-percent or greater and (2) the object comprised at least 5-percent of the total area of the image.

Other object types, including pedestrians, cyclists, and buses were ignored. The bounding boxes obtained from the object detector were used to "crop" each street-level image to isolate each car. We then run our image classifier on each of the isolated car images. 

#### Features of UCF Google StreetView Dataset:

62,058 Google Street View images covering areas of Pittsburgh, Orlando, partially Manhattan.

Each image includes accurate GPS coordinates and compass direction.

Images cover a variety of weather, season, and time of day.

### 3. Estimate Price and Map Predictions

The average value (MSRP in United States from TrueCar.com for more current models, and estimated values for old models using listings on AutoTrader.com) of each of the car manufacturers was obtained. A new class, "Car Class" was created by breaking the car values into ranges and assigning each car manufacturer a class. The classes include Economy, Standard, Luxury and Exotic/Super Luxury. 

The goal of this visual census was to estimate the relative socio-economic status of each neighborhood traversed in our street-level data. To do so, we counted the number of each car class in each scene. 