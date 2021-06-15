from joblib import load, dump
import pandas as pd
import re
import numpy as np


df_2016 = load('../datasets/Peninsula/df_pen_wind_2016_1_0.pkl')
col = df_2016.columns
ini_lat, fin_lat = 35.5, 44.0 
ini_lon, fin_lon = -9.5, 4.5
step = 0.5
listLat = list(np.arange(ini_lat, fin_lat+step, step))
listLon = list(np.arange(ini_lon, fin_lon+step, step))
search_coord = []
for lat in listLat:
    for lon in listLon:
        search_coord.append(f'({lat}, {lon})')
new_coord = []
for coord in search_coord:
    r = re.compile(f".*{coord}")
    new_coord += list(filter(r.match, col))
newDataframe = df_2016[new_coord]
newDataframe.index = df_2016.index

newDataframe.to_csv('../datasets/df_pen_wind_2016_reduced.csv')
#fullDataset.dump('../datasets/df_pen_wind_2016_2018.pkl')