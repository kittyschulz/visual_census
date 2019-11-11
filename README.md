# Visual Census Using Car-Type Image Classifier on Street-Level Imagery

The goal of this work is to use a fine-grained car model image classifier and object detector on geo-localized images to determine if they can provide meaningful census signals.

We build a fine-grained image classifier by fine-tuning a ResNet152 image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). A pre-trained object detector from the [TF-Hub module](https://www.tensorflow.org/hub/overview) is used to extract car-type objects from street view data. The car type objects are classified by our fine-grained classification model. 

## 1. Fine-Grained Car Classification

The first step in performing this Visual Census is to fine tune a pre-trained image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). 

### The Stanford Cars Dataset

![stanford](/img/stanford_sample.png)

The dataset is comprised of a total of 16,185 images with 196 unique labels of car make-model-year. The data is split into a 8,144-image training set and 8,041-image testing set. There is an average of 41 distinct images per label in the training set, which include multiple points of view for each car label.

The car models contained in the dataset were all avliable on the North American market. The models ranged from cars like the [Geo Metro](https://en.wikipedia.org/wiki/Geo_Metro) to the [Bugatti Veyron](https://en.wikipedia.org/wiki/Bugatti_Veyron). Model years ranged from 1991 to 2012. 

The 196 labels include cars from 49 different manufacturers. Note that the representation of manufacturers in the dataset is not representative of [U.S. marketshare](https://www.statista.com/statistics/249375/us-market-share-of-selected-automobile-manufacturers/). For instance, the Stanford Cars Dataset contains a disproportionate amount of exotic or ultra-luxury cars (cars costing more than $200,000USD). Of the 8,144 images in the training set, 1,072 can be classified as 'exotic' or 'ultra-luxury' cars (about 13 percent). 

![mnfr_counts](/img/mnfr_hist.png)

### Image Pre-Processing

To fine-tune a classifier on the Stanford dataset, each image was cropped, resized, and normalized using ``` pre-process.py ```. The images were cropped based on the bounding boxes provided in the training set and resized to 224 by 224 pixels. Each image was padded with a 16-pixel margin. The pixels were normalized to values between 0 and 1.

### Initial Models

Initial benchmark models were built using the Keras Sequential model API. The Keras Sequential API is a faster alternative to other, better-performing models, allowing for a great number of trials in a short time. 

In addition to the original 196 labels of the Stanford Cars Dataset, the benchmark models were also trained on and used to predict average car values and car class. The performance of the benchmark model was superior using the Car Value and Car Class labels, with a maximum validation accuracy of 66 percent for Vehcile Type and 68 percent for Value. The validation accuracy of the Keras Sequential model on all 196 original labels was a meager 21 percent.

### ResNet152 Image Classifier

The final model is a ResNet152 model pretrained on ImageNet and fine-tuned on the Stanford Cars Dataset. The structure of the model follows that of the original Caffe implementation. The weights for the pre-trained model can be [downloaded](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view) from Google Drive. 

The model obtained a final validation accuracy of 88.8 percent on the validation set. 

Below is a sample of 16 images from the test set. Fifteen of the sixteen vehicles have been classified correcetly, with the exception of the Ferrari GTC in the bottom row which our model predicted to be a Jaguar. 

![classified cars](/img/classified_cars.png)

## 2. Object Detection on Street-Level Scenes

The car-type objects were detected in scenes of the [UCF Google Streetview data](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset) through the use of a [TF-Hub module](https://www.tensorflow.org/hub/overview) trained to perform object detection. Each car-type object detected, with a confidence of at least 25 percent, we run through the image classifier. 

![obj detect](/img/obj_detector.png)

Other object types, including pedestrians, cyclists, and buses were ignored. The bounding boxes obtained from the object detector were used to "crop" each street-level image to isolate each car. We then ran our image classifier on each of the isolated car-type objects. 

The scenes had a median count of 12 vehicles and a maximum of 39 vehicles per scene. Five percent of the scenes contained less than five vehicles. 

![num cars](/img/number_of_cars.png)

### The UCF Google StreetView Dataset

![streetview](/img/street_vew_sample.png)

The UCF Google StreetView Dataset is comprised of 62,058 Google Street View images covering areas of Pittsburgh, Orlando, and Manhattan for 10,343 individual scenes. There are five (5) views per scene: four side views, one sky view, one repeated side view with annotations. Each image includes GPS coordinates (latitude and longitude) and compass direction.

Of the 10,343 scenes, xx take place in Pittsburgh, xx in Orlando, and xx in Manhattan. 

[ bar graph of scene count ]

## 3. Mapping Predictions

We have visualized 

%[![Foo](http://www.google.com.au/images/nav_logo7.png)](http://google.com.au/)

## Conclusions


# Implementation

This work can be reproduced by cloning this repository and following along below. We recommend running an instance on Google Cloud using the [Deep Learning VM](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) with at least one GPU. 

### Dependencies

To reproduce this work, the following packages must be installed:

For loading, writing, and saving data:
- pickle
- json

For manipulating data and images:
- Pandas
- Numpy
- Pillow

For building and training our model:
- Console-Progressbar
- OpenCV2
- TensorFlow2 and Keras
- Scipy

For constructing our maps:
- Folium
- Webbrowser

### Helpful Tips

 - If you are running on an instance, remember to call ``` $ nohup ``` so that these scripts will run in the background, and continue running even if you lose connection to your VM.

- Monitor your GPU throughout this project by calling ``` $ watch nvidia-smi ```. As always, you can also keep track of your CPU and Memory utilization with ``` $ top ``` or ``` $ htop ```.

### Image Classifier

To get the Stanford Cars Data, call the following in your terminal:

```
$ cd visual_census
$ mkdir ucf_data
$ cd ucf_data
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

The data will be downloaded as *.tgz files. To extract and pre-process the data, call:

 ``` $ python3 pre-process.py ```

With the images pre-processed, we can train the model. First, [download](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view)  the weights for the pre-trained ResNet152 model. 

The trained model will be saved as an *.hdf5 file. If you don't have access to a machine with multiple GPUs or a cluster, it is recommended you let this step run overnight. 

```
$ python3 build_model.py
```

### Object Detector

To get the UCF Street View Data, call the following in your terminal. You can expect this step will take up to 6 hours to complete.

```
$ cd ../object_detector
$ python3 get_ucf_data.py
```

The data will be extracted from the zipped format and stored in the ``` ucf_data ``` folder. Therefore, the object detector can immediately be run on the images by calling:

 ``` $ python3 ucf_detector.py ``` 
 
 By default, the object detector will only store car-type objects that have a confidence above 25 percent.

The output of ``` ucf_detector.py ``` will be a pickled dictionary containing bounding box coordinates and a confidence for each of the car-type objects. This dictionary will be used to crop the images in the pre-processing step when we run our image classifier. 

### Using the Model

Now that the image classifier is trained and the car-type objects have been identified in the street view scenes, we can classify the street view vehicles and use these predictions to create some nice visualizations. 

Run the image classifier on the car-type objects detected. You can expect this step to run for 2 to 3 hours on the UCF Street View Data.

```
$ python3 image_classifier/test_streetview.py
```

The output of ``` test_streetview.py ``` will be a json file containing a predicted label for each object, the object's scene number (the first 6 characters of the image file name from which it was detected), and the probability assigned to the label. Of course, the label stored for each object is the label predicted with the highest probability. 

Using the *.json file of the prediction results, we can build our first map. Do so by calling:

``` $ python3 maps/build_folium_map.py ```

We'll construct a heatmap using [Folium](https://python-visualization.github.io/folium/) of the luxury cars observed along the SteetView Car's route. You can easily construct your own map by editing ``` build_folium_map.py ``` and exploring the Folium documentation.

## Limitations and Future Work

It is important to be mindful of the current limitations of this work. These are items that we hope to address in the future, but for the time being should be kept in mind when exploring the model predictions.

#### Limitations:

- We did not account for yellow cabs, police cars, or commercial vehicles. Such vehicles are, of course, numerous on city streets and their presence may be interesting to account for when examining different apects of a neighborhood's socio-economic status.  

- Parked cars and moving cars are given the same weight in our prediction. Parked cars indicate a resident or vistor to the neighborhood, whereas moving vehicles may just be passing through. Depending on what we wish to explore from this data, this differentiation may be important. 

- Our image classifier was trained on a pristine dataset, and only obtained a validation accuracy of 88.8 percent. Although our validation accuracy was good, it was nowhere near state of the art, nor is the accuracy this high on real-world data from the Street View dataset. This means that some of the predictions we make on the StreetView data is wrong, and almost certainly, the proportion of incorrect predictions is more than 11.2 percent. 

- The UCF StreetView dataset only traversed a few neighborhoods in Manhattan, and the very small downtown area of Pittsburgh and Orlando. Especially in the Manhattan portion of the dataset, many neighborhoods of lower socio-economic status were excluded, making it difficult for us to map perceptible disparities from our predictions. 

#### Future Work:

- We will be expanding the StreetView data to cover more of Manhattan and Pittsburgh using the Google StreetLearn dataset. Please stay tuned!

- The Stanford Cars Dataset used to train the model will be supplemented with additional vehicles. We will focus on adding models manufactuered after 2012, and adding additional images of the most common makes and models observed in the real-world data.

- Improve the accuracy of the image classifier closer to the current state-of-the-art for Stanford Cars Dataset.

- Continue exploring relationships between different socio-economic factors and vehicles and traffic.

Thank you for taking the time to explore this project with us. We welcome your comments, advice, and suggestions! Please contact Katerina Schulz at katerina.schulz [at] gatech.edu
