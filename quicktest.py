import numpy as np
import pandas as pd
import math
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from nltk.stem.porter import *
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cross_validation import train_test_split

path = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot'
final_xtrain = path + '/src/data_res/xtrain.csv'
final_xtest = path + '/src/data_res/xtest.csv'
final_ytrain = path + '/src/data_res/ytrain.csv'
final_id_test = path + '/src/data_res/id_test.csv'


fname_tfidf = path + '/data/result_tfidf.csv'
result_tfidf = pd.read_csv(fname_tfidf, sep=',', na_values='(MISSING)', encoding="utf-8")

fname_new_feature = path + '/data/new_feature.csv'
result_new_feature = pd.read_csv(fname_new_feature, sep=',', na_values='(MISSING)', encoding="utf-8")

df_frame = pd.concat([result_tfidf, result_new_feature], axis=1)
y_train = pd.read_csv(final_ytrain, sep=',', na_values='(MISSING)', encoding="utf-8")

y_train = y_train['relevance']
train_row = y_train.shape[0]


X_train = df_frame[:train_row]
X_train.to_csv(final_xtrain, index=False)
X_test = df_frame[train_row:]
X_test.to_csv(final_xtest, index=False)

X_train = pd.read_csv(final_xtrain, sep=',', na_values='(MISSING)', encoding="utf-8")

a_train, a_test, b_train, b_test = train_test_split(X_train, y_train, test_size=0.33)
print 'done'
params = {u'colsample_bylevel': 0.93,
     u'colsample_bytree': 0.65,
     u'gamma': 0.75,
     u'learning_rate': 0.022000000000000002,
     u'max_depth': 12,
     u'min_child_weight': 0.0,
     u'n_estimators': 1035,
     u'nthread': -1,
     u'objective': u'reg:linear',
     u'reg_alpha': 0.755,
     u'reg_lambda': 4.58,
     u'seed': 43,
     u'silent': 1,
    u'subsample': 0.9}
gbm = xgb.XGBRegressor(**params).fit(a_train, b_train, eval_metric="rmse",early_stopping_rounds=100, eval_set=[(a_test, b_test)])