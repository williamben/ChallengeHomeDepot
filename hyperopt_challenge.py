import numpy as np
import pandas as pd
import math
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
from nltk.stem.porter import *
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cross_validation import train_test_split
import itertools

path = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot'
final_xtrain = path + '/src/data_res/xtrain.csv'
final_xtest = path + '/src/data_res/xtest.csv'
final_ytrain = path + '/src/data_res/ytrain.csv'
final_id_test = path + '/src/data_res/id_test.csv'


fname_tfidf = path + '/data/result_tfidf.csv'
result_tfidf = pd.read_csv(fname_tfidf, sep=',', na_values='(MISSING)', encoding="utf-8")

fname_new_feature = path + '/data/new_feature.csv'
result_new_feature = pd.read_csv(fname_new_feature, sep=',', na_values='(MISSING)', encoding="utf-8")

fname_data_counting = path + '/data/data_counting.csv'
result_data_counting = pd.read_csv(fname_data_counting, sep=',', na_values='(MISSING)', encoding="utf-8")
result_data_counting = result_data_counting._get_numeric_data()


y_train = pd.read_csv(final_ytrain, sep=',', na_values='(MISSING)', encoding="utf-8")

y_train = y_train['relevance']
train_row = y_train.shape[0]

df_data_counting = result_data_counting._get_numeric_data()
col_to_remove = []
for pair in itertools.combinations(df_data_counting.columns,2):
    if all(df_data_counting[pair[0]] == df_data_counting[pair[1]]):
        if pair[1] not in col_to_remove:
            col_to_remove.append(pair[1])

df_data_counting.drop(col_to_remove, inplace=True, axis=1)

df_frame = pd.concat([result_tfidf, result_new_feature, df_data_counting], axis=1)
X_train = df_frame[:train_row]
X_train.to_csv(final_xtrain, index=False)
X_test = df_frame[train_row:]
X_test.to_csv(final_xtest, index=False)

X_train = pd.read_csv(final_xtrain, sep=',', na_values='(MISSING)', encoding="utf-8")

a_train, a_test, b_train, b_test = train_test_split(X_train, y_train, test_size=0.33)
print 'done'


space4rf = {

            'colsample_bytree': hp.quniform('colsample_bytree', 0.05, 0.9, 0.05),
            'gamma': hp.quniform('gamma', 0., 10., 0.05),
            'learning_rate': hp.quniform('eta', 0.0001, 0.04, 0.0005),
            'max_depth': hp.choice('max_depth', range(4, 20)),
            'min_child_weight': hp.quniform('min_child_weight', 0., 6., 1),
            'n_estimators': 3500,
            'nthread': -1,
            'objective': 'reg:linear',
            'reg_alpha': hp.quniform('reg_alpha', 0.0, 10.0, 0.005),
            'reg_lambda': hp.quniform('reg_lambda', 0.0, 10.0, 0.005),
            'subsample': hp.quniform('subsample', 0.8, 1, 0.05),
            'colsample_bylevel': hp.quniform('colsample_bylevel', 0.4, 1, 0.005),
            'silent': 1,
            'seed': 43
             }

best = 0
df_result_hyperopt = pd.DataFrame(columns=[np.append(['score', 'best iter'], (space4rf.keys()))])
i = 0


def rmse_score(params):
    global i

    print '------------------'
    print i
    print params


    gbm = xgb.XGBRegressor(**params).fit(a_train, b_train, eval_metric="rmse",early_stopping_rounds=100, eval_set=[(a_test, b_test)])

    y_pred_train = gbm.predict(a_test, ntree_limit=gbm.best_iteration)
    y_pred_train[y_pred_train < 1] = 1
    y_pred_train[y_pred_train > 3] = 3
    loss = math.sqrt(MSE(b_test, y_pred_train))
    df_result_hyperopt.columns = [np.append(['score','best_iter'], (params.keys()))]
    df_result_hyperopt.loc[i] = np.append([loss, int(gbm.best_iteration)], params.values())


    print 'last: ', loss

    print '------------------'
    #print 'done'
    i = i+1
    print 'next'
    return {'loss': loss, 'status': STATUS_OK}

trials = Trials()
best = fmin(rmse_score, space4rf, algo=tpe.suggest, max_evals=40, trials=trials)
print 'best:'
print best

df_result_hyperopt['n_estimators'] = df_result_hyperopt['best_iter']
del df_result_hyperopt['best_iter']
df_result_hyperopt = df_result_hyperopt.sort_values(
    'score', axis=0, ascending=[True])

fname_hyperopt = path + '/data/result_hyperopt.csv'
df_result_hyperopt.to_csv(fname_hyperopt, index=False)

