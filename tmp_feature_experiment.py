import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from scipy.optimize import minimize

ROOT = Path(r'c:\Users\hp\OneDrive\Desktop\Gridlock')
train = pd.read_csv(ROOT/'train.csv')
TARGET='demand'
BASE32='0123456789bcdefghjkmnpqrstuvwxyz'

def decode_geohash(value):
    lat=[-90.0,90.0]; lon=[-180.0,180.0]; even=True
    for char in str(value).lower():
        num = BASE32.index(char)
        for mask in (16,8,4,2,1):
            interval = lon if even else lat
            mid=(interval[0]+interval[1])/2
            if num & mask:
                interval[0]=mid
            else:
                interval[1]=mid
            even = not even
    return (lat[0]+lat[1])/2,(lon[0]+lon[1])/2

def build_features(df):
    data=df.copy()
    tp=data['timestamp'].str.split(':',expand=True).astype(int)
    data['hour']=tp[0]
    data['minute']=tp[1]
    data['minutes_since_midnight']=data['hour']*60+data['minute']
    data['quarter']=data['minutes_since_midnight']//15
    data['day_index']=data['day']-48
    data['dayofweek']=(data['day']-1)%7
    data['weekend']=data['dayofweek'].isin([5,6]).astype(int)
    data['rush_hour']=data['hour'].isin([7,8,9,16,17,18,19]).astype(int)
    data['night_flag']=((data['hour']<6)|(data['hour']>=22)).astype(int)
    data['hour_sin']=np.sin(2*np.pi*data['minutes_since_midnight']/1440)
    data['hour_cos']=np.cos(2*np.pi*data['minutes_since_midnight']/1440)
    unique_geohashes=data['geohash'].dropna().unique()
    decoded={gh:decode_geohash(gh) for gh in unique_geohashes}
    data['geohash_lat']=data['geohash'].map(lambda x:decoded.get(x,(0.0,0.0))[0])
    data['geohash_lon']=data['geohash'].map(lambda x:decoded.get(x,(0.0,0.0))[1])
    data['geohash_prefix4']=data['geohash'].str[:4]
    data['geohash_prefix5']=data['geohash'].str[:5]
    data['temp_missing']=data['Temperature'].isna().astype(int)
    data['weather_missing']=data['Weather'].isna().astype(int)
    return data

features=build_features(train.drop(columns=[TARGET]))
train_feat=features.copy()
train_feat['Temperature']=train_feat['Temperature'].fillna(train_feat['Temperature'].median())
for c in ['RoadType','Weather','LargeVehicles','Landmarks']:
    train_feat[c]=train_feat[c].fillna('Missing').astype(str)
train_feat['demand']=train[TARGET].values
prev=train[['geohash','day','timestamp','demand']].copy()
prev['day']+=1
prev.rename(columns={'demand':'prev_day_demand'}, inplace=True)
train_feat=train_feat.merge(prev,on=['geohash','day','timestamp'],how='left')
train_feat['prev_day_missing']=train_feat['prev_day_demand'].isna().astype(int)

valid_idx=train_feat[train_feat['day']==49].sample(frac=0.2, random_state=42).index
train_idx=train_feat.index.difference(valid_idx)
train_stats=train_feat.loc[train_idx]
for name, cols in [
    ('gh_ts_mean',['geohash','timestamp']),
    ('gh_mean',['geohash']),
    ('ts_mean',['timestamp']),
    ('gh4_mean',['geohash_prefix4']),
    ('gh5_mean',['geohash_prefix5']),
    ('road_mean',['RoadType']),
    ('weather_mean',['Weather']),
    ('gh_hour_mean',['geohash','hour']),
    ('gh_quarter_mean',['geohash','quarter']),
]:
    grp=train_stats.groupby(cols)['demand'].mean()
    train_feat[name]=train_feat.set_index(cols).index.map(grp).astype(float).fillna(train_stats['demand'].mean())

feature_cols=['day_index','hour','minute','minutes_since_midnight','quarter','dayofweek','weekend','rush_hour','night_flag','hour_sin','hour_cos',
              'geohash_lat','geohash_lon','Temperature','NumberofLanes','prev_day_missing','prev_day_demand','temp_missing','weather_missing',
              'geohash','geohash_prefix4','geohash_prefix5','RoadType','LargeVehicles','Landmarks','Weather','timestamp',
              'gh_mean','ts_mean','gh_ts_mean','gh4_mean','gh5_mean','road_mean','weather_mean','gh_hour_mean','gh_quarter_mean']
cat_cols=['geohash','geohash_prefix4','geohash_prefix5','RoadType','LargeVehicles','Landmarks','Weather','timestamp']

cb = CatBoostRegressor(iterations=900,learning_rate=0.035,depth=6,loss_function='RMSE',random_seed=42,verbose=False,thread_count=-1,allow_writing_files=False)
cb.fit(train_feat.loc[train_idx, feature_cols], train_feat.loc[train_idx,'demand'], cat_features=cat_cols)
cb_pred=cb.predict(train_feat.loc[valid_idx,feature_cols])
print('cat r2', r2_score(train_feat.loc[valid_idx,'demand'], cb_pred))

corr_train = train_feat.loc[train_idx][train_feat.loc[train_idx,'prev_day_missing']==0]
corr_valid = train_feat.loc[valid_idx][train_feat.loc[valid_idx,'prev_day_missing']==0]
print('corr sizes', len(corr_train), len(corr_valid))
if len(corr_valid)>0:
    corr_model = CatBoostRegressor(iterations=650,learning_rate=0.04,depth=6,loss_function='RMSE',random_seed=42,verbose=False,thread_count=-1,allow_writing_files=False)
    corr_model.fit(corr_train[feature_cols], corr_train['demand'], cat_features=cat_cols)
    corr_pred=corr_model.predict(corr_valid[feature_cols])
    print('corr r2', r2_score(corr_valid['demand'], corr_pred))
    base_valid_matched = cb.predict(corr_valid[feature_cols])
    combined = np.column_stack([base_valid_matched, corr_pred])
    def objective(w): return -r2_score(corr_valid['demand'], combined@w)
    res = minimize(objective, [0.5,0.5], bounds=[(0,1),(0,1)], constraints={'type':'eq','fun':lambda w:w.sum()-1})
    print('weights', res.x, 'blend r2', r2_score(corr_valid['demand'], combined@res.x))
