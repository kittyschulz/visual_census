# Visual Census Using Car-Type Image Classifier on Street-Level Imagery

The goal of this work is to use a fine-grained car model image classifier and object detector on geo-localized images to determine if they can provide meaningful census signals.

We build a fine-grained image classifier by fine-tuning a ResNet152 image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). A pre-trained object detector from the [TF-Hub module](https://www.tensorflow.org/hub/overview) is used to extract car-type objects from street view data. The car type objects are classified by our fine-grained classification model. 

## 1. Fine-Grained Car Classification

The first step in performing this Visual Census is to fine tune a pre-trained image classifier on the [Stanford cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). 

### The Stanford Cars Dataset

![stanford](/img/stanford_sample.png)

The dataset is comprised of a total of 16,185 images with 196 unique labels of car make-model-year. The data is split into a 8,144-image training set and 8,041-image testing set. There is an average of 41 distinct images per label in the training set, which include multiple points of view for each car label.

The car models contained in the dataset were all available on the North American market. The models ranged from cars like the [Geo Metro](https://en.wikipedia.org/wiki/Geo_Metro) to the [Bugatti Veyron](https://en.wikipedia.org/wiki/Bugatti_Veyron). Model years ranged from 1991 to 2012. 

The 196 labels include cars from 49 different manufacturers. Note that the representation of manufacturers in the dataset is not representative of [U.S. marketshare](https://www.statista.com/statistics/249375/us-market-share-of-selected-automobile-manufacturers/). For instance, the Stanford Cars Dataset contains a disproportionate amount of exotic or ultra-luxury cars (cars costing more than $200,000USD). Of the 8,144 images in the training set, 1,072 can be classified as 'exotic' or 'ultra-luxury' cars (about 13 percent). 

<p align="center">
<img align="center" src="/img/mnfr_hist.png">
</p>

### Image Pre-Processing

To fine-tune a classifier on the Stanford dataset, each image was cropped, resized, and normalized using ``` pre-process.py ```. The images were cropped based on the bounding boxes provided in the training set and resized to 224 by 224 pixels. Each image was padded with a 16-pixel margin. The pixels were normalized to values between 0 and 1.

### Initial Models

Initial benchmark models were built using the Keras Sequential model API. The Keras Sequential API is a faster alternative to other models such as ResNet50, VGG16, or ResNet152, allowing for a great number of trials in a short time. 

In addition to the original 196 labels of the Stanford Cars Dataset, the benchmark models were also trained on and used to predict car value class and car type class, metrics which had a total of four (4) and nine (9) labels, respectively. The performance of the benchmark model was superior when using the Car Value and Car Class labels, with a maximum validation accuracy of 66 percent for Vehicle Type and 68 percent for Value. The validation accuracy of the Keras Sequential model on all 196 original labels was a meager 21 percent. 

### ResNet152 Image Classifier

A low validation accuracy when using the Keras Sequential models on a fine-grained problem comes as no surprise. Fine-grained classification is a notoriously difficult problem, and requires a much more robust model to perform well. Therefore, the final model is a ResNet152 model pre-trained on ImageNet and fine-tuned on the Stanford Cars Dataset. The structure of the model follows that of the original Caffe implementation. The weights for the pre-trained model can be [downloaded](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view) from Google Drive. 

The model obtained a final validation accuracy of 88.8 percent on the validation set. 

Below is a sample of 16 images from the test set. Fifteen of the sixteen vehicles have been classified correctly, with the exception of the Ferrari GTC in the bottom row which our model predicted to be a Jaguar. 

<p align="center">
<img align="center" src="/img/classified_cars.png">
</p>

## 2. Object Detection on Street-Level Scenes

The car-type objects were detected in scenes of the [UCF Google Streetview data](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset) through the use of a [TF-Hub module](https://www.tensorflow.org/hub/overview) trained to perform object detection. Each car-type object detected, with a confidence of at least 25 percent, is run through the image classifier. 

<p align="center">
<img align="center" src="/img/obj_detector.png">
</p>

The scenes had a median count of 12 vehicles and a maximum of 39 vehicles per scene. Five percent of the scenes contained less than five vehicles. 

Other object types, including pedestrians, cyclists, and buses were ignored. The bounding boxes obtained from the object detector were used to "crop" each street-level image to isolate each car. We then ran our image classifier on each of the isolated car-type objects. 

### The UCF Google StreetView Dataset

<p align="center">
<img align="center" src="/img/street_vew_sample.png">
</p>

The UCF Google StreetView Dataset is comprised of 62,058 Google Street View images covering areas of Pittsburgh, Orlando, and Manhattan for 10,343 individual scenes. There are five (5) views per scene: four side views, one sky view, one repeated side view with annotations. Each image includes GPS coordinates (latitude and longitude) and compass direction.

Of the 10,343 scenes, 5,941 take place in Pittsburgh, 1,324 in Orlando, and 3,078 in Manhattan. 

<p align="center">
<img align="center" src="/img/scenes_city.png">
</p>

## 3. Mapping Predictions

We visualize the predictions of the image classifier on real-world scenes using [Folium](https://python-visualization.github.io/folium/). Below is a heatmap of luxury cars throughout a portion of Midtown Manhattan.

[![folium_snippet](/img/folium_snippet.png)](http://kittyschulz.github.io/map.html)

Additional interactive maps are available on the [project website](http://kittyschulz.github.io/map.html). 

## Conclusions

We wish to draw meaningful census signals from the classification of car-type objects in real-world scenes. Thus far, we have used the models created in this work to explore the ratio of [luxury vehicles](http://kittyschulz.github.io/map.html) throughout cityscapes, [traffic counts](http://kittyschulz.github.io/map.html) in city centers, and distribution of vehicles by [age](http://kittyschulz.github.io/map.html), [class](http://kittyschulz.github.io/map.html), and approximated [value](http://kittyschulz.github.io/map.html). 

We know that each of these can be tied to [different socio-economic factors](https://www.pnas.org/content/114/50/13108), however drawing conclusions from our data goes beyond the scope of this work, and requires validation from additional census data. 

# Implementation

This work can be reproduced by cloning this repository and following the instructions below. We recommend running an instance on Google Cloud using the [Deep Learning VM](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) with at least one GPU. 

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

 - If you are running on an instance, remember to call ``` $ nohup ``` when calling each script so that the process will run in the background, and continue running even if you lose connection to your instance.

- Monitor your GPU throughout this project by calling ``` $ watch nvidia-smi ```. As always, you can also keep track of CPU and Memory utilization with ``` $ top ``` or ``` $ htop ```.

### Image Classifier

Assuming you have already cloned this repository, to get the Stanford Cars Data, call the following in your terminal:

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

With the images pre-processed, we can train the model. First, [download](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view) the weights for the pre-trained ResNet152 model. 

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

In ``` test_streeview.py ``` change the name of the ``` model_weights_path ``` to the name of the model output by training your image classifier. It will be located in the ``` models ``` folder. Run the image classifier on the car-type objects detected. You can expect this step to run for 2 to 3 hours on the UCF Street View Data.

```
$ python3 image_classifier/test_streetview.py
```

The output of ``` test_streetview.py ``` will be a json file containing a predicted label for each object, the object's scene number (the first 6 characters of the image file name from which it was detected), and the probability assigned to the label. Of course, the label stored for each object is the label predicted with the highest probability. 

Using the *.json file of the prediction results, we can build our first map. Do so by calling:

``` $ python3 maps/mapping.py ```

This will construct a heatmap using [Folium](https://python-visualization.github.io/folium/) of the luxury cars observed along the SteetView Car's route. You can easily construct your own map by editing ``` mapping.py ``` and exploring the Folium documentation.

# Limitations and Future Work

It is important to be mindful of the current limitations of this work. These are items that we hope to address in the future, but for the time being should be acknowledged when exploring the model predictions.

#### Limitations:

- We did not account for yellow cabs, police cars, or commercial vehicles. Such vehicles are, of course, numerous on city streets and their presence may be interesting to account for when examining different aspects of a neighborhood's socio-economic status.  

- Parked cars and moving cars are given the same weight in our prediction. Parked cars indicate a resident or visitor to the neighborhood, whereas moving vehicles may just be passing through. Depending on what we wish to explore from this data, this differentiation may be important. 

- Our image classifier was trained on a pristine dataset, and only obtained a validation accuracy of 88.8 percent. Although our validation accuracy was good, it was nowhere near state of the art, nor is the accuracy this high on real-world data from the Street View dataset. This means that some of the predictions we make on the StreetView data is wrong, and almost certainly, the proportion of incorrect predictions is more than 11.2 percent. 

- The UCF StreetView dataset only traversed a few neighborhoods in Manhattan, and the very small downtown area of Pittsburgh and Orlando. Especially in the Manhattan portion of the dataset, many neighborhoods of lower socio-economic status were excluded, making it difficult for us to map perceptible disparities from our predictions. 

#### Future Work:

- We will be expanding the StreetView data to cover more of Manhattan and Pittsburgh using the Google StreetLearn dataset. Please stay tuned!

- The Stanford Cars Dataset used to train the model will be supplemented with additional vehicles. We will focus on adding models manufactured after 2012, and adding additional images of the most common makes and models observed in the real-world data.

- Improve the accuracy of the image classifier closer to the current state-of-the-art for Stanford Cars Dataset.

- Continue exploring relationships between different socio-economic factors and vehicles and traffic.

Thank you for taking the time to explore this project with us. We welcome your comments, advice, and suggestions! Please contact Katerina Schulz at katerina.schulz [at] gatech.edu

# References

**3D Object Representations for Fine-Grained Categorization**  
Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei  
*4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013* (**3dRR-13**). Sydney, Australia. Dec. 8, 2013.


**Image Geo-localization Based on Multiple Nearest Neighbor Feature Matching using Generalized Graphs**  
Amir Roshan Zamir and Mubarak Shah  
*IEEE Transactions on Pattern Analysis and Machine Intelligence* (**TPAMI**), 2014

