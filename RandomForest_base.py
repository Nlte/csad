from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

def _timestamp_to_float(t):
    return float(time.mktime(t.to_datetime().timetuple()))

DATA_DIR = "project_data/week_lon_lat_bs"

# Train (7 first weeks)
train_files = ["exo1_b3_week%s.csv" % i for i in range(1,7)]
train_files = [os.path.join(DATA_DIR, f) for f in train_files]
list_ = []
for f in train_files:
    df_ = pd.read_csv(f, names=["year", "month", "day", "hour", "in_id",
        "in_long", "in_lat", "out_id", "out_long", "out_lat", "calls", "durations"])
    list_.append(df_)
df = pd.concat(list_)
calls = df["calls"]
df = df.drop(["calls", "durations", "year", "in_long", "in_lat", "out_long", "out_lat"], axis = 1)

# Test (last week)
test_file = os.path.join(DATA_DIR, "exo1_b3_week8.csv")
df_test = pd.read_csv(test_file, names=["year", "month", "day", "hour", "in_id",
    "in_long", "in_lat", "out_id", "out_long", "out_lat", "calls", "durations"])
df_test = df_test.drop(["durations", "year", "in_long", "in_lat", "out_long", "out_lat"], axis = 1)


minimum_calls = np.min(calls)
maximum_calls = np.max(calls)
mean_calls = np.mean(calls)
median_calls = np.median(calls)
std_calls = np.std(calls)

# stats
print("Statistics for BS:\n")
print("Minimum calls: %f" % minimum_calls)
print("Maximum calls: %f" % maximum_calls)
print("Mean calls: %f" % mean_calls)
print("Median calls %f" % median_calls)
print("Standard deviation of calls: %f" % std_calls)


from sklearn.metrics import mean_squared_error

def mse_metric(y_true, y_predict):
    score = mean_squared_error(y_true, y_predict)
    return score

from sklearn.model_selection import train_test_split

# convert timestamp to float
#df['date'] = pd.to_datetime(df['date'])
#df['date_delta'] = (df['date'] - df['date'].min()) / np.timedelta64(1,'D')
#df = df.drop("date", axis=1)


print("Splitting df into train and test datasets.")
X_train = df
y_train = calls
X_test = df_test[["month", "day", "hour", "in_id", "out_id"]]
y_test = df_test["calls"]
#X_train, X_test, y_train, y_test = train_test_split(df, calls, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def fit_DTR(X, y):
    rs = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state=0)
    cv_sets = rs.get_n_splits(X.shape[0])
    reg = DecisionTreeRegressor()
    params = {'max_depth': range(1,5)}
    scoring_fn = make_scorer(mse_metric)
    grid = GridSearchCV(reg, params, scoring=scoring_fn, cv=cv_sets)
    grid = grid.fit(X, y)
    opt_model = grid.best_estimator_
    print("DTR : optimal depth {}".format(opt_model.get_params()['max_depth']))
    return opt_model


from sklearn.ensemble.forest import RandomForestRegressor
def fit_RFR(X, y):
    rs = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state=0)
    cv_sets = rs.get_n_splits(X.shape[0])
    reg = RandomForestRegressor()
    params = {'n_estimators': range(1,5)}
    scoring_fn = make_scorer(mse_metric)
    grid = GridSearchCV(reg, params, scoring=scoring_fn, cv=cv_sets)
    grid = grid.fit(X, y)
    opt_model = grid.best_estimator_
    print("RFR : optimal num estimators {}".format(opt_model.get_params()['n_estimators']))
    return opt_model

# Fitting
print("Fitting DTR")
reg_DTR = fit_DTR(X_train, y_train)
print("Fitting RFR")
reg_RFR = fit_RFR(X_train, y_train)


predictions_DTR = reg_DTR.predict(X_test)
predictions_RFR = reg_RFR.predict(X_test)
print("DTR score %f" % mean_squared_error(y_test, predictions_DTR))
print("RFR score %f" % mean_squared_error(y_test, predictions_RFR))

# Plot
df_test["predictions_DTR"] = pd.Series(predictions_DTR)
df_test["predictions_RFR"] = pd.Series(predictions_RFR)
df_dtr = df_test[["calls", "predictions_DTR"]].groupby([df_test.hour]).sum()
df_dtr.plot()
df_rfr = df_test[["calls", "predictions_RFR"]].groupby([df_test.hour]).sum()
df_rfr.plot()
plt.show()
