### A Visual Census Using Car-Type Detector on Street-Level Imagery
1. Fine-Grained Car Classification

The Stanford Cars dataset contains a total of 196 unique cars over approximately 16,000 images. The car models contained in the dataset all originate from the North American market.

The first step in performing this Visual Census is to train a pre-built image classifier to predict the make (*and model) of vehicles. The image classifier will be fine-tuned on the Stanford Cars dataset. To train the classifier on the Stanford dataset, each image must have its dimensions normalized. 

The labels for each image are contained in 'cars_annos.mat', a MatLab file. To extract the image labels and metadata, scipy.io was used to convert the *.mat file to a dictionary from which the desired data could be extracted to a Pandas DataFrame.

2. Object Detection on Street-Level Scenes

A pre-built object detector was used to identify the car-type objects in each scene of the Mapillary street-level data. Other object types, including pedestrians, cyclists, and buses were ignored. The bounding boxes obtained from the object detector were used to "crop" each street-level image to isolate each car. 

We then run our image classifier on each of the isolated car images. 

3. Estimate Price and Map Predictions

The cost (MSRP in United States [link to source]) of each of the 196 car models was obtained and mapped to the predictions. 

The goal of this visual census was to estimate the socio-economic status of each neighborhood traversed in our street-level data. To do so, we obtained the median cost over all vehicles per scene, as well as the cost of the most expensive car in each scene. 

A heat map was constructed based on the geo-location data provided in the Mapillary dataset for each image. 