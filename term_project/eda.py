from sklearn import metrics, ensemble
from sklearn.model_selection import train_test_split, GridSearchCV

# import xgboost as xgb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

train_new = pd.read_csv('./temporal_data/train_id_cnt_svd_stamp_before_after.csv')
test_new = pd.read_csv('./temporal_data/test_id_cnt_svd_stamp_before_after.csv')
members_new = pd.read_csv('./temporal_data/members_id_cnt_svd_stamp.csv')
songs_new = pd.read_csv('./temporal_data/songs_id_cnt_svd_stamp.csv')

songs_drop = songs_new.drop(songs_new.columns[23:71],axis=1)
songs_drop = songs_drop.drop(songs_drop.columns[1:23],axis=1)
songs_drop = songs_drop.drop(songs_drop.columns[17:19],axis=1)
songs_drop.info()

members_drop = members_new.drop(members_new.columns[101:119], axis=1)
members_drop = members_drop.drop(members_drop.columns[1:53],axis=1)
members_drop.info()

train_drop = train_new.drop(train_new.columns[6:46], axis=1)
train_drop = train_drop.drop(train_drop.columns[2:5], axis=1)
train_drop.info()

train_tmp = pd.merge(train_drop, songs_drop, on='song_id')
train_tmp.shape

ct = pd.crosstab(train_tmp.artist_component_0, train_new.target)
ct.plot.bar(figsize=(100, 100), stacked=False)
plt.savefig('./artist_component_0_target.png')
plt.show()