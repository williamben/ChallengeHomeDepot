import pandas as pd
import numpy as np
import xgboost as xgb
import time

path = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot'
final_xtrain = path + '/src/data_res/xtrain.csv'
final_xtest = path + '/src/data_res/xtest.csv'
final_ytrain = path + '/src/data_res/ytrain.csv'
final_id_test = path + '/src/data_res/id_test.csv'


X_train = pd.read_csv(final_xtrain, sep=',', na_values='(MISSING)', encoding="utf-8")
y_train = pd.read_csv(final_ytrain, sep=',', na_values='(MISSING)', encoding="utf-8")
X_test = pd.read_csv(final_xtest, sep=',', na_values='(MISSING)', encoding="utf-8")
id_test = pd.read_csv(final_id_test, sep=',', na_values='(MISSING)', encoding="utf-8")
fname_hyperopt = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot/data/result_hyperopt.csv'
resultat_hyperopt = pd.read_csv(fname_hyperopt, sep=',', na_values='(MISSING)', encoding="utf-8")


def create_submission(resultat_hyperopt, nombre):
    y_pred_res = np.zeros(X_test.shape[0])
    for i in xrange(nombre):
        best_param = resultat_hyperopt.sort_values('score', ascending=True).head(nombre).to_dict(orient='records')[i]
        print best_param
        del best_param['score']
        best_param['n_estimators'] = int(best_param['n_estimators'])
        params = best_param
        params['nthread'] = -1
        gbm = xgb.XGBRegressor(**params)
        gbm.fit(X_train, y_train, eval_metric="rmse")
        y_pred = gbm.predict(X_test)
        y_pred[y_pred < 1] = 1
        y_pred[y_pred > 3] = 3
        y_pred_res += y_pred
    return y_pred_res/float(nombre)

y_pred_final = create_submission(resultat_hyperopt, 1)
timestr = time.strftime("%m%d-%H%M")
fname_sub = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot/submission/submission'+timestr+'.csv'
pd.DataFrame({"id": id_test['id'], "relevance": y_pred_final}).to_csv(fname_sub,index=False)
print 'done'