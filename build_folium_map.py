# for loading the object data
import pickle
import json

# for constructing our map
import pandas as pd
import numpy as np
import scipy.io
import folium

# for viewing our map
import os
import webbrowser

def load_data(json_path, gps_data_path, filter_exotics=True):
    """
    Loads pickled dictionary output from object
    detector and matlab file containing latitude and
    longitude of scenes. Creates pandas DataFrame to
    use for building a map in Folium.
    
    Args:
        pickle_path (str): path to a pickled dict of 
        results from the object detector
    
        gps_data_path (str): path to gps data for scenes

        filter_exotics (bool): filtering instances class-
        -ified as exotic cars will reduce outliers and 
        improve the map if the folium map is constructed
        using value attributes

    Returns:
        pandas DataFrame with index of scene name and 
        colums of lat, long, compass orientation and 
        object counts.
    """
    # with open(pickle_path, 'rb') as f:
    #     ucf_objects_mobilenet = pickle.load(f)

    # object_counts = {}
    # for key, value in ucf_objects_mobilenet.items():
    #     object_counts[key] = len(value)

    with open(json_path, "r") as read_file:
        prediction_data = json.load(read_file)

    prediction_data = pd.DataFrame(prediction_data)
    
    prediction_data['scene'] = prediction_data['scene'].map(lambda x: x.split('_')[0])
    prediction_data['year'] = prediction_data['label'].map(lambda x: int(x.split(' ')[-1]))

    image_names = ['0'*(6-len(str(num))) + str(num) for num in range(1,10344)]
    gps = np.squeeze(scipy.io.loadmat(gps_data_path)['GPS_Compass'])
    gps_df = pd.DataFrame(gps, index=image_names, columns=['lat', 'long', 'compass'])

    all_scene_data = prediction_data.merge(gps_df, how='outer', left_on='scene', right_index=True)

    # all_views_obj_count = {}
    # for key, value in object_counts.items():
    #     if key[:6] in all_views_obj_count:
    #         all_views_obj_count[key[:6]] += value
    #     else:
    #         all_views_obj_count[key[:6]] = value

    # gps_df['count'] = gps_df.index.map(all_views_obj_count)

    return all_scene_data

def construct_map(city, scene_data, scene_data_luxury, attribute, path_save):
    """
    Constructs an interactive map using Folium 
    and a pandas DataFrame containing latitude
    and longitude of scenes and scene attributes
    (for example, object counts).

    Args:
        city (tuple): A tuple of latitude and 
        longitude corrdinates of the desired 
        city to center the map on.

        scene_data (pandas DataFrame): DataFrame
        containing scene information including
        latitude, longitude, compass orientation
        and other scene attributes (such as object
        instance counts).

        attribute (str): The desired scene attribute 
        on which to build the map.

        path_save (str): The path and file name to 
        save the map to.

    Returns:
        None
    """

    # scene_data['marker_color'] = pd.cut(scene_data[attribute], bins=5, 
    #                             labels=['red', 'orange', 'yellow', 'green', 'blue'])

    # scene_data['radius'] = pd.cut(scene_data[attribute], bins=5, 
    #                             labels=[5, 4, 3, 2, 1])
    
    # create empty map zoomed in on Manhattan
    map = folium.Map(location=city, zoom_start=12)
    
    # add a marker for every record in the data #each[1]['radius'], each[1]['marker_color']
    for each in scene_data.iterrows():
        folium.CircleMarker(location = [each[1]['lat'], each[1]['long']], radius=2, color='black', opacity=1).add_to(map)

    for each in scene_data_luxury.iterrows():
        folium.CircleMarker(location = [each[1]['lat'], each[1]['long']], radius=1, color='red', opacity=1).add_to(map)

    map.save(path_save)

def main():
    # get paths to json file with preditctions, scene data, and where map should be saved
    json_path='/Users/katerina/Workspace/visual_census/results_resnet_v2.json'
    gps_data_path='ucf_data/gps/GPS_Long_Lat_Compass.mat'
    path_save='maps/map.html'

    scene_data = load_data(json_path, gps_data_path)
    # scene_data = scene_data.groupby('scene').mean()
    luxury =['Audi', 'BMW', 'Land Rover', 'Lexus', 'Mercedes-Benz', 'Porsche', 'Tesla', 'Volvo']
    scene_data_luxury = scene_data[scene_data['label'].str.contains('|'.join(luxury), na=False)]

    attribute='year'
    NY_COORDINATES = (40.7831, -73.9712)
    construct_map(NY_COORDINATES, scene_data, scene_data_luxury, attribute, path_save)

    # automatically loads the map in the default browser
    webbrowser.open('file://' + os.path.realpath(path_save))

if __name__ == "__main__":
    main()