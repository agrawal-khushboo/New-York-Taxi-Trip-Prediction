import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.decomposition import PCA

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Trip duration clean-up
m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2*s]
train = train[train['trip_duration'] >= m - 2*s]

# Latitude and longitude clean-up
train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

# pca = PCA().fit(coords)
# train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
# train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
# train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
# train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
# test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
# test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
# test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
# test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday
train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear
train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute
train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']
test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday
test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear
test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute
test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']

train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 2)
train.loc[:, 'center_lat_bin'] = np.round(train['center_latitude'], 2)
train.loc[:, 'center_long_bin'] = np.round(train['center_longitude'], 2)
train.loc[:, 'pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))
test.loc[:, 'pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)
test.loc[:, 'pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
test.loc[:, 'center_lat_bin'] = np.round(test['center_latitude'], 2)
test.loc[:, 'center_long_bin'] = np.round(test['center_longitude'], 2)
test.loc[:, 'pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

train.loc[:, 'centroid_pick_lat'] = kmeans.cluster_centers_[train['pickup_cluster']][:,0]
train.loc[:, 'centroid_pick_long'] = kmeans.cluster_centers_[train['pickup_cluster']][:,1]
train.loc[:, 'centroid_drop_lat'] = kmeans.cluster_centers_[train['dropoff_cluster']][:,0]
train.loc[:, 'centroid_drop_long'] = kmeans.cluster_centers_[train['dropoff_cluster']][:,1]
train.loc[:, 'cluster_haversine'] = haversine_array(train['centroid_pick_lat'].values, train['centroid_pick_long'].values, train['centroid_drop_lat'].values, train['centroid_drop_long'].values)
train.loc[:, 'cluster_manhtn'] = dummy_manhattan_distance(train['centroid_pick_lat'].values, train['centroid_pick_long'].values, train['centroid_drop_lat'].values, train['centroid_drop_long'].values)
train.loc[:, 'cluster_bearing'] = bearing_array(train['centroid_pick_lat'].values, train['centroid_pick_long'].values, train['centroid_drop_lat'].values, train['centroid_drop_long'].values)

test.loc[:, 'centroid_pick_lat'] = kmeans.cluster_centers_[test['pickup_cluster']][:,0]
test.loc[:, 'centroid_pick_long'] = kmeans.cluster_centers_[test['pickup_cluster']][:,1]
test.loc[:, 'centroid_drop_lat'] = kmeans.cluster_centers_[test['dropoff_cluster']][:,0]
test.loc[:, 'centroid_drop_long'] = kmeans.cluster_centers_[test['dropoff_cluster']][:,1]
test.loc[:, 'cluster_haversine'] = haversine_array(test['centroid_pick_lat'].values, test['centroid_pick_long'].values, test['centroid_drop_lat'].values, test['centroid_drop_long'].values)
test.loc[:, 'cluster_manhtn'] = dummy_manhattan_distance(test['centroid_pick_lat'].values, test['centroid_pick_long'].values, test['centroid_drop_lat'].values, test['centroid_drop_long'].values)
test.loc[:, 'cluster_bearing'] = bearing_array(test['centroid_pick_lat'].values, test['centroid_pick_long'].values, test['centroid_drop_lat'].values, test['centroid_drop_long'].values)

# N = 100000
# city_long_border = (-74.03, -73.75)
# city_lat_border = (40.63, 40.85)
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], s=10, lw=0,
#            c=train.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)
# ax.set_xlim(city_long_border)
# ax.set_ylim(city_lat_border)
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# plt.show()

fr1 = pd.read_csv('fastest_routes_train_part_1.csv',
                  usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv('fastest_routes_train_part_2.csv',
                  usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv('fastest_routes_test.csv',
                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')

for gby_col in ['pickup_hour', 'pickup_date', 'pickup_dt_bin',
               'pickup_week_hour', 'pickup_cluster', 'dropoff_cluster']:
    gby = train.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m']]
    gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
    train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
    test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

for gby_cols in [['center_lat_bin', 'center_long_bin'],
                 ['pickup_hour', 'center_lat_bin', 'center_long_bin'],
                 ['pickup_hour', 'pickup_cluster'],  ['pickup_hour', 'dropoff_cluster'],
                 ['pickup_cluster', 'dropoff_cluster']]:
    coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
    coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
    coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
    # coord_stats = coord_stats[coord_stats['id'] > 100]
    coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]
    train = pd.merge(train, coord_stats, how='left', on=gby_cols)
    test = pd.merge(test, coord_stats, how='left', on=gby_cols)

group_freq = '60min'
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

# Count trips over 60min
df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']
train = train.merge(df_counts, on='id', how='left')
test = test.merge(df_counts, on='id', how='left')

# Count how many trips are going to each cluster over time
dropoff_counts = df_all \
    .set_index('pickup_datetime') \
    .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster']) \
    .agg({'id': 'count'}) \
    .reset_index().set_index('pickup_datetime') \
    .groupby('dropoff_cluster').rolling('240min').mean() \
    .drop('dropoff_cluster', axis=1) \
    .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
    .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

train['dropoff_cluster_count'] = train[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)
test['dropoff_cluster_count'] = test[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)

# Count how many trips are going from each cluster over time
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
pickup_counts = df_all \
    .set_index('pickup_datetime') \
    .groupby([pd.TimeGrouper(group_freq), 'pickup_cluster']) \
    .agg({'id': 'count'}) \
    .reset_index().set_index('pickup_datetime') \
    .groupby('pickup_cluster').rolling('240min').mean() \
    .drop('pickup_cluster', axis=1) \
    .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
    .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_cluster_count'})

train['pickup_cluster_count'] = train[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
test['pickup_cluster_count'] = test[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)

# feature_names = list(train.columns)
do_not_use_for_training = ['id', 'pickup_datetime', 'dropoff_datetime', 'pickup_dt', 'pickup_week_hour',
                           'trip_duration', 'avg_speed_h_gby_pickup_dt_bin', 'avg_speed_m_gby_pickup_dt_bin',
                           'pickup_date', 'avg_speed_h', 'avg_speed_m', 'avg_speed_h_center_lat_bin_center_long_bin',
                           'pickup_lat_bin', 'pickup_long_bin', 'cnt_center_lat_bin_center_long_bin',
                           'center_lat_bin', 'center_long_bin',
                           'pickup_dt_bin', 'pickup_datetime_group',
                           'avg_speed_h_gby_pickup_hour', 'avg_speed_m_gby_pickup_hour', 'pickup_minute',
                           'centroid_pick_lat', 'centroid_pick_long', 'centroid_drop_lat', 'centroid_drop_long']
# feature_names = [f for f in train.columns if f not in do_not_use_for_training]

# print train.count()

### TRAIN DATA ONLY ###

df = train['total_distance'].isnull()
idxToDelete = df[df==True].index.values.astype(int)
train.drop(idxToDelete, inplace=True)

# categorical_columns = ['pickup_weekday', 'pickup_hour', 'pickup_cluster', 'dropoff_cluster', 'pickup_hour_weekofyear']
# for cat_col in categorical_columns:
# 	one_hot = pd.get_dummies(train[cat_col],drop_first=True)
# 	one_hot_cols = one_hot.columns
# 	one_hot.columns = [cat_col+str(a) for a in one_hot_cols]
# 	train = train.drop(cat_col,axis = 1)
# 	train = train.join(one_hot)

y = np.log(train['trip_duration'].values + 1)
train = train.drop(columns=do_not_use_for_training)

do_not_use_for_testing = ['id', 'pickup_datetime', 'pickup_dt', 'pickup_week_hour',
                           'avg_speed_h_gby_pickup_dt_bin', 'avg_speed_m_gby_pickup_dt_bin',
                           'pickup_date', 'avg_speed_h_center_lat_bin_center_long_bin',
                           'pickup_lat_bin', 'pickup_long_bin', 'cnt_center_lat_bin_center_long_bin',
                           'center_lat_bin', 'center_long_bin',
                           'pickup_dt_bin', 'pickup_datetime_group',
                           'avg_speed_h_gby_pickup_hour', 'avg_speed_m_gby_pickup_hour', 'pickup_minute',
                           'centroid_pick_lat', 'centroid_pick_long', 'centroid_drop_lat', 'centroid_drop_long']

# print train.count()

# print train.columns
Xtr, Xv, ytr, yv = train_test_split(train.values, y, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_pars = {'min_child_weight': 10, 'eta': 0.11, 'colsample_bytree': 0.5, 'max_depth': 15,
            'subsample': 0.9, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 200, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=10)
print('Modeling RMSLE %.5f' % model.best_score)

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

feature_names = [f for f in train.columns]
print len(feature_names)
ceate_feature_map(feature_names)

impt = model.get_fscore(fmap='xgb.fmap')
importance = impt.items()
# importance = []
# for i in range(len(impt)):
#   importance.append((feature_names[i], impt[i][1]))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 12))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()

# FOREVER_COMPUTING_FLAG = True
# xgb_pars = []
# for MCW in [10]:
#     for ETA in [0.1, 0.11, 0.125]:
#         for CS in [0.4, 0.45, 0.5]:
#             for MD in [14, 15]:
#                 for SS in [0.85, 0.9]:
#                     for LAMBDA in [1, 1.2, 1.35, 1.5]:
#                         xgb_pars.append({'min_child_weight': MCW, 'eta': ETA, 
#                                          'colsample_bytree': CS, 'max_depth': MD,
#                                          'subsample': SS, 'lambda': LAMBDA, 
#                                          'nthread': 8, 'booster' : 'gbtree', 'eval_metric': 'rmse',
#                                          'silent': 1, 'objective': 'reg:linear'})

# while FOREVER_COMPUTING_FLAG:
#     xgb_par = np.random.choice(xgb_pars, 1)[0]
#     print(xgb_par)
#     model = xgb.train(xgb_par, dtrain, 100, watchlist, early_stopping_rounds=50,
#                       maximize=False, verbose_eval=10)
#     print('Modeling RMSLE %.5f' % model.best_score)

### Kaggle Test Data ###
# feature_names = [f for f in test.columns if f not in do_not_use_for_testing]
# dtest = xgb.DMatrix(test[feature_names].values)
# ytest = model.predict(dtest)
# test['trip_duration'] = np.exp(ytest) - 1
# test[['id', 'trip_duration']].to_csv('anmol_xgb_submission.csv.gz', index=False, compression='gzip')