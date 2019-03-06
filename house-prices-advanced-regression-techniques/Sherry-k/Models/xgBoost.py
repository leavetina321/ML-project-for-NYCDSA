#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:53:35 2019

@author: shehryarkhawaja
"""

import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
import numpy  as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
%matplotlib inline
import pickle as pk
import xgboost as xgb


raw = pd.read_csv("./data/traindata_full_dummified.csv")
raw_test = pd.read_csv("./data/testdata_full_dummified.csv")
print(raw.shape)
print(raw_test.shape)

##find outliers and take them out
def find_outliers(df,col):
    mean = np.mean(df[col], axis=0)
    sd = np.std(df[col], axis=0)
    gap=mean-4*sd
    gap2=mean+4*sd
    outliers = df[ (df[col] > gap2) | (df[col]< gap)].index
    return outliers


miscval=list(find_outliers(raw, 'MiscVal'))
lotarea=list(find_outliers(raw, 'LotArea'))
droplist=miscval+lotarea
raw.drop(droplist,axis=0, inplace=True)


raw1 = raw.copy()
raw1 = raw1.drop('Id', axis = 1)
sale_price = raw['LogSalePrice']
raw = raw.drop(['LogSalePrice','Id'],axis=1)
X = raw.copy()
Y = sale_price.copy()
test_IDs = pd.DataFrame(raw_test['Id'])
raw_test = raw_test.drop('Id', axis = 1)


print(X.shape)
print(Y.shape)
print(raw1.shape)

##creating train test split for initial model fit
#train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=0)


##find outliers and take them out
def find_outliers(df,col):
    mean = np.mean(df[col], axis=0)
    sd = np.std(df[col], axis=0)
    gap=mean-4*sd
    gap2=mean+4*sd
    outliers = df[ (df[col] > gap2) | (df[col]< gap)].index
    return outliers


xgboost = xgb.XGBRegressor()
xgboost.fit(X, Y)
print("Score: {0:}".format(xgboost.score(X, Y)))
print("Mean squared error: {0:}".format(mean_squared_error(Y, xgboost.predict(X))))






### Running CV to find best parameters for model

params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(50, 250), # default 100
    "subsample": uniform(0.6, 0.4)}

rand_search = RandomizedSearchCV(xgboost, param_distributions=params, random_state=0, n_iter=200, cv = 3, verbose=1, n_jobs=1, return_train_score=True)

## model to print out results and return params of CV
def get_CV_results(model):
    model.fit(X,Y)
    (score,i) = max((v,i) for i,v in enumerate(model.cv_results_['mean_test_score']))
    print("Mean validation score: {0:.3f} (std: {1:.4f})".format(score,
        model.cv_results_['std_test_score'][i]))
    print("Mean train score: {0:.3f} (std: {1:.5f})".format(
        model.cv_results_['mean_train_score'][i],
        model.cv_results_['std_train_score'][i]))
    print("Params:\n{})".format(model.cv_results_['params'][i]))
    return model.cv_results_['params'][i]

rand_search_params = get_CV_results(rand_search)



### setting model params for from searches
xgboost_tuned = xgb.XGBRegressor(**rand_search_params)


##funtion to run the model and print its results
def run_model(model):
    model.fit(X, Y)
    print("Score: {0:}".format(model.score(X, Y)))
    print("Mean squared error: {0:}".format(mean_squared_error(Y, model.predict(X))))
    print(filename)
    results = pd.DataFrame(np.exp(model.predict(raw_test)))
    predictions = pd.concat([test_IDs,results], axis = 1)
    predictions.columns = ['Id', 'SalePrice']
    return predictions

##running model 
run_model(xgboost_tuned)
xgboost_tuned.score(X, Y)

#train_ID = raw1['Id']
#results_train = pd.DataFrame(np.exp(xgboost_tuned.predict(X)))
#predictions_train = pd.concat([train_ID,results_train], axis = 1)
#predictions_train.columns = ['Id', 'SalePrice']
#predictions_train.to_csv('xgBoost_train_predict.csv', index = False)

results = pd.DataFrame(np.exp(xgboost_tuned.predict(raw_test)))
predictions = pd.concat([test_IDs,results], axis = 1)
predictions.columns = ['Id', 'SalePrice']
predictions.to_csv('xgBoost_predict_new.csv', index = False)
