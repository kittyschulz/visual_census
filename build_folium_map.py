# for loading the object data
import pickle

# for constructing our map
import pandas as pd
import numpy as np
import scipy.io
import folium

# for viewing our map
import os
import webbrowser

def get_gps_data(pickle_path, gps_data_path):
    """
    Loads pickled dictionary output from object
    detector and matlab file containing latitude and
    longitude of scenes. Creates pandas DataFrame to
    use for building a map in Folium.
    
    Args:
        pickle_path (str): path to a pickled dict of 
        results from the object detector
    
        gps_data_path (str): path to gps data for scenes

    Returns:
        pandas DataFrame with index of scene name and 
        colums of lat, long, compass orientation and 
        object counts.
    """
    with open(pickle_path, 'rb') as f:
        ucf_objects_mobilenet = pickle.load(f)

    object_counts = {}
    for key, value in ucf_objects_mobilenet.items():
        object_counts[key] = len(value)

    image_names = ['0'*(6-len(str(num))) + str(num) for num in range(1,10344)]

    gps = np.squeeze(scipy.io.loadmat(gps_data_path)['GPS_Compass'])
    gps_df = pd.DataFrame(gps, index=image_names, columns=['lat', 'long', 'compass'])

    all_views_obj_count = {}
    for key, value in object_counts.items():
        if key[:6] in all_views_obj_count:
            all_views_obj_count[key[:6]] += value
        else:
            all_views_obj_count[key[:6]] = value

    gps_df['count'] = gps_df.index.map(all_views_obj_count)

    return gps_df

def construct_map(city, gps_data, attribute, path_save):
    """
    Constructs an interactive map using Folium 
    and a pandas DataFrame containing latitude
    and longitude of scenes and scene attributes
    (for example, object counts).

    Args:
        city (tuple): A tuple of latitude and 
        longitude corrdinates of the desired 
        city to center the map on.

        gps_data (pandas DataFrame): DataFrame
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

    gps_data['marker_color'] = pd.cut(gps_data[attribute], bins=5, 
                                labels=['red', 'orange', 'yellow', 'green', 'blue'])
    
    # create empty map zoomed in on Manhattan
    map = folium.Map(location=city, zoom_start=12)
    
    # add a marker for every record in the data
    for each in gps_data.iterrows():
        folium.CircleMarker(location = [each[1]['lat'], each[1]['long']], radius=each[1][attribute]*0.25, color=each[1]['marker_color'], opacity=0.25).add_to(map)

    map.save(path_save)

def main():
    # get paths to pickle, scene data, and where map should be saved
    pickle_path='object_detector/ucf_objects_detected_mobilenet.pickle'
    gps_data_path='ucf_data/gps/GPS_Long_Lat_Compass.mat'
    path_save='maps/map.html'

    attribute='count'
    NY_COORDINATES = (40.7831, -73.9712)

    gps_data = get_gps_data(pickle_path, gps_data_path)
    construct_map(NY_COORDINATES, gps_data, attribute, path_save)

    # automatically loads the map in the default browser
    webbrowser.open('file://' + os.path.realpath(path_save))

if __name__ == "__main__":
    main()