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

def construct_map(gps_data, path_save):
    NY_COORDINATES = (40.7831, -73.9712)

    gps_data['marker_color'] = pd.cut(gps_data['count'], bins=5, 
                                labels=['red', 'orange', 'yellow', 'green', 'blue'])
    
    # create empty map zoomed in on San Francisco
    map = folium.Map(location=NY_COORDINATES, zoom_start=12)
    
    # add a marker for every record in the filtered data, use a clustered view
    for each in gps_data.iterrows():
        folium.CircleMarker(location = [each[1]['lat'], each[1]['long']], radius=each[1]['count']*0.25, color=each[1]['marker_color'], opacity=0.25).add_to(map)

    map.save(path_save)

def main():
    pickle_path='ucf_objects_detected_mobilenet.pickle'
    gps_data_path='/Users/katerina/Workspace/visual_census/ucf_data/gps/GPS_Long_Lat_Compass.mat'
    path_save='/Users/katerina/Workspace/visual_census/map.html'

    gps_data = get_gps_data(pickle_path, gps_data_path)
    construct_map(gps_data, path_save)

    webbrowser.open('file://' + os.path.realpath(path_save))

if __name__ == "__main__":
    main()