import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

train = pd.read_csv('train.csv')

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

y = np.log(train['trip_duration'].values + 1)
# y = train['trip_duration'].values
Xtr, Xv, ytr, yv = train_test_split(train.values, y, test_size=0.2, random_state=1987)

pred = np.mean(ytr)

error_2 = (pred-yv)*(pred-yv)
# error_2 = (np.log(pred+1)-np.log(yv+1))*(np.log(pred+1)-np.log(yv+1))
mse = np.mean(error_2)
rmse = np.sqrt(mse)

print "Validation RMSLE = ", rmse

plt.hist(y, bins=100)
plt.xlabel('log(trip_duration+1)')
plt.ylabel('number of train records')
plt.show()