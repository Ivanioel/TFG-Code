import pandas as pd
import numpy as np

df_2016 = pd.read_csv('../datasets/Sotavento/data_target_stv_2016.csv', index_col=0, parse_dates=True)
df_2017 = pd.read_csv('../datasets/Sotavento/data_target_stv_2017.csv', index_col=0, parse_dates=True)

df_2016_2017 = pd.concat([df_2016, df_2017])
df_2016_2017.to_csv('../datasets/Sotavento/data_target_stv_2016_2017.csv')
