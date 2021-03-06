# for loading the object data
import pickle
import json

# for constructing our map
import pandas as pd
import numpy as np
import scipy.io
import folium
from folium.plugins import HeatMap

# for viewing our map
import os
import webbrowser


def load_data(json_path, gps_data_path, filter_exotics=True):
    """
    Loads pickled dictionary output from object detector and matlab file 
    containing latitude and longitude of scenes. Creates pandas DataFrame 
    to use for building a map in Folium.

    Args:
        pickle_path (str): path to a pickled dict of results from the object detector

        gps_data_path (str): path to gps data for scenes

        filter_exotics (bool): filtering instances classified as exotic cars will 
        reduce outliers and improve the map if the folium map is constructed using 
        value attributes

    Returns:
        pandas DataFrame with index of scene name and colums of lat, long, compass 
        orientation and object counts.
    """
    # with open(pickle_path, 'rb') as f:
    #     ucf_objects_mobilenet = pickle.load(f)

    # object_counts = {}
    # for key, value in ucf_objects_mobilenet.items():
    #     object_counts[key] = len(value)

    with open(json_path, "r") as read_file:
        prediction_data = json.load(read_file)

    prediction_data = pd.DataFrame(prediction_data)

    prediction_data['scene'] = prediction_data['scene'].map(
        lambda x: x.split('_')[0])
    prediction_data['year'] = prediction_data['label'].map(
        lambda x: int(x.split(' ')[-1]))

    image_names = ['0'*(6-len(str(num))) + str(num) for num in range(1, 10344)]
    gps = np.squeeze(scipy.io.loadmat(gps_data_path)['GPS_Compass'])
    gps_df = pd.DataFrame(gps, index=image_names, columns=[
                          'lat', 'long', 'compass'])

    all_scene_data = prediction_data.merge(
        gps_df, how='outer', left_on='scene', right_index=True)

    # all_views_obj_count = {}
    # for key, value in object_counts.items():
    #     if key[:6] in all_views_obj_count:
    #         all_views_obj_count[key[:6]] += value
    #     else:
    #         all_views_obj_count[key[:6]] = value

    # gps_df['count'] = gps_df.index.map(all_views_obj_count)

    return all_scene_data


def construct_map(city, data, path_save):
    """
    Constructs an interactive map using Folium and a pandas DataFrame containing latitude
    and longitude of scenes and scene attributes (for example, object counts).

    Args:
        city (tuple): A tuple of latitude and longitude corrdinates of the desired city 
        to center the map on.

        scene_data (pandas DataFrame): DataFrame containing scene information including
        latitude, longitude, compass orientation and other scene attributes (such as object
        instance counts).

        attribute (str): The desired scene attribute on which to build the map.

        path_save (str): The path and file name to save the map to.

    Returns:
        None
    """

    gps_data = data.groupby(['scene']).mean()

    gps_data['marker_color'] = pd.cut(gps_data['year'], bins=5,
                                labels=['red', 'orange', 'yellow', 'green', 'blue'])

    gps_data['radius'] = pd.cut(gps_data['year'], bins=5,
                                labels=[5, 4, 3, 2, 1])

    # create empty map zoomed in on Manhattan
    map = folium.Map(location=city, zoom_start=12)

    # add a marker for every record in the data #each[1]['radius'], each[1]['marker_color']

    for each in gps_data.iterrows():
        folium.CircleMarker(location=[
                            each[1]['lat'], each[1]['long']], radius=each[1]['radius'], color=each[1]['marker_color'], opacity=0.5).add_to(map)

    # luxury = ['Chevrolet Corvette ZR1 2012', 'Honda Accord Coupe 2012', 'Mercedes-Benz S-Class Sedan 2012', 'Jaguar XK XKR 2012', 'BMW X6 SUV 2012', 'BMW X5 SUV 2007', 'Volvo C30 Hatchback 2012', 'Audi S6 Sedan 2011', 'BMW X3 SUV 2012', 'Land Rover Range Rover SUV 2012', 'BMW M3 Coupe 2012', 'Mercedes-Benz E-Class Sedan 2012',
    #           'Chevrolet Corvette Convertible 2012', 'Land Rover LR2 SUV 2012', 'Audi S5 Coupe 2012', 'Tesla Model S Sedan 2012', 'Audi R8 Coupe 2012', 'BMW M5 Sedan 2010', 'Mercedes-Benz SL-Class Coupe 2009', 'Porsche Panamera Sedan 2012', 'BMW M6 Convertible 2010', 'Audi S5 Convertible 2012', 'BMW ActiveHybrid 5 Sedan 2012', 'Audi A5 Coupe 2012', 'Audi S4 Sedan 2012']

    # Take only the most popular luxury cars
    # heat_df = data[data['label'].str.contains('|'.join(luxury), na=False)]
    # heat_df = heat_df[['lat', 'long']]
    # heat_df = heat_df.dropna(axis=0, subset=['lat', 'long'])

    # # List comprehension to make out list of lists
    # heat_data = [[row['lat'], row['long']]
    #             for index, row in heat_df.iterrows()]

    # HeatMap(heat_data).add_to(map)

    # for each in scene_data_luxury.iterrows():
    #     folium.CircleMarker(location=[
    #                         each[1]['lat'], each[1]['long']], radius=1, color='red', opacity=1).add_to(map)

    map.save(path_save)


def main():
    # get paths to json file with preditctions, scene data, and where map should be saved
    json_path = '/Users/katerina/Workspace/visual_census/results_resnet_v2.json'
    gps_data_path = '/Users/katerina/Workspace/visual_census/ucf_data/gps/GPS_Long_Lat_Compass.mat'
    path_save = '/Users/katerina/Workspace/visual_census/maps/vehicle-age.html'

    scene_data = load_data(json_path, gps_data_path)
    #scene_data = scene_data.groupby('scene')
    #luxury = ['Land Rover', 'Mercedes-Benz', 'Porsche', 'Tesla', 'Volvo']
    # scene_data_luxury = scene_data[scene_data['label'].str.contains(
    #     '|'.join(luxury), na=False)]

    NY_COORDINATES = (40.7831, -73.9712)
    construct_map(NY_COORDINATES, scene_data, path_save)

    # automatically loads the map in the default browser
    webbrowser.open('file://' + os.path.realpath(path_save))


if __name__ == "__main__":
    main()
