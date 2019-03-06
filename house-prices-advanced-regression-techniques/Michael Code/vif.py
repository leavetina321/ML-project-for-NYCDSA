#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:16:41 2019

@author: michaelsankari
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy  as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from sklearn import linear_model, metrics
from sklearn.linear_model import Ridge, Lasso, LinearRegression
plt.style.use('ggplot')
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import datetime
import pickle

def get_stack(X, Y):
        
    ridge_file = 'vif ridge2019-02-28 18/10/09.790796.sav'
    ridge_file = ridge_file.replace('/', ':') #python is looking for colons, not slashes
    ridge = pickle.load(open(ridge_file, 'rb'))
    
    linear_file = 'vif linear2019-02-28 18/10/09.871399.sav'
    linear_file = linear_file.replace('/', ':') #python is looking for colons, not slashes
    linear = pickle.load(open(linear_file, 'rb'))
    
    final_model = LinearRegression()
    
    final_model.intercept_ = (ridge.intercept_ + linear.intercept_)/2
    final_model.coef_ = (ridge.coef_ + linear.coef_)/2
    
    final_model.predict(X)
    final_model.score(X, Y)
    
    suffix = str(datetime.datetime.now())
    model_filename = 'vif stack' + suffix +'.sav'
    pickle.dump(linear, open(model_filename, 'wb'))
    csv_filename = 'vif stack ' + suffix + '.csv'
    
    raw_test, test_IDs = load_test()
    predict = final_model.predict(raw_test)
    predict = np.exp(predict)
    predict = pd.DataFrame(predict)
    predict = pd.concat([test_IDs, predict], axis = 1)
    predict.columns = ['Id', 'SalePrice']
    predict.to_csv(csv_filename, index=False)

def get_linear(X, Y, vif):
    linear = LinearRegression()
    linear.fit(X, Y)
    
    suffix = str(datetime.datetime.now())
    model_filename = 'vif linear' + suffix +'.sav'
    pickle.dump(linear, open(model_filename, 'wb'))
    csv_filename = 'vif linear ' + suffix + '.csv'
    
    raw_test, test_IDs = load_test()
    predict = linear.predict(raw_test)
    predict = np.exp(predict)
    predict = pd.DataFrame(predict)
    predict = pd.concat([test_IDs, predict], axis = 1)
    predict.columns = ['Id', 'SalePrice']
    predict.to_csv(csv_filename, index=False)

def get_ridge(X, Y, vif):
    #running ridge with alpha 0 (MLR)
    ridge = Ridge()
    ridge.set_params(alpha = 0, normalize = True)
    ridge.fit(X, Y)
    
    #Grid search for ridge
    alphas_ridge = np.linspace(0,15,50)
    tuned_parameters_r = [{'alpha': alphas_ridge}]
    n_folds = 5
    cv = KFold(n_splits=n_folds, shuffle=True)
    
    tune_ridge = GridSearchCV(ridge, tuned_parameters_r, 
                              cv=cv, refit=True, return_train_score = True, 
                              scoring = 'neg_mean_squared_error')
    tune_ridge.fit(X,Y)
    
    print(tune_ridge.best_params_)
    print(np.max(tune_ridge.cv_results_['mean_test_score']))
    print(np.min(tune_ridge.cv_results_['mean_test_score']))
    
    ridge_best = tune_ridge.best_estimator_
    ridge_best.fit(X,Y)
    print(ridge_best.score(X,Y))
    
    suffix = str(datetime.datetime.now())
    model_filename = 'vif ridge' + suffix +'.sav'
    pickle.dump(ridge_best, open(model_filename, 'wb'))
    csv_filename = 'vif ridge ' + suffix + '.csv'
    
    raw_test, test_IDs = load_test()
    predict = ridge_best.predict(raw_test)
    predict = np.exp(predict)
    predict = pd.DataFrame(predict)
    predict = pd.concat([test_IDs, predict], axis = 1)
    predict.columns = ['Id', 'SalePrice']
    predict.to_csv(csv_filename, index=False)


def get_lasso(X, Y, vif):
    #running ridge with alpha 0 (MLR)
    lasso = Lasso()
    lasso.set_params(alpha = 0, normalize = True)
    lasso.fit(X, Y)
    
    #Grid search for ridge
    alphas_lasso = np.linspace(0,10,50)
    tuned_parameters_r = [{'alpha': alphas_lasso}]
    n_folds = 5
    cv = KFold(n_splits=n_folds, shuffle=True)
    
    tune_lasso = GridSearchCV(lasso, tuned_parameters_r, 
                              cv=cv, refit=True, return_train_score = True, 
                              scoring = 'neg_mean_squared_error')
    tune_lasso.fit(X,Y)
    
    print(tune_lasso.best_params_)
    print(np.max(tune_lasso.cv_results_['mean_test_score']))
    print(np.min(tune_lasso.cv_results_['mean_test_score']))
    
    lasso_best = tune_lasso.best_estimator_
    lasso_best.fit(X,Y)
    print(lasso_best.score(X,Y))

    suffix = str(datetime.datetime.now())
    model_filename = 'vif lasso' + suffix +'.sav'
    pickle.dump(lasso_best, open(model_filename, 'wb'))
    csv_filename = 'vif ridge ' + suffix + '.csv'
    
    raw_test, test_IDs = load_test()
    predict = lasso_best.predict(raw_test)
    predict = np.exp(predict)
    predict = pd.DataFrame(predict)
    predict = pd.concat([test_IDs, predict], axis = 1)
    predict.columns = ['Id', 'SalePrice']
    predict.to_csv(csv_filename, index=False)

def load_train():
    '''
    Reads in the dummified data set. Does further cleaning to drop unnecessary columns
    and does transformations on price.
    Returns raw, the final data set for the features and sale_price, the final dependent variable
    '''
    os.chdir('/Users/michaelsankari/Documents/NYC Data Science/Machine Learning Project/github/ML-project-for-NYCDSA')
    raw = pd.read_csv("./house-prices-advanced-regression-techniques/cleandata/s2_clean_dummified.csv")
    
    outliers = raw[ (raw['GrLivArea'] > 4000) & (raw['LogSalePrice'] < 13) ].index
    raw.drop(outliers,axis=0, inplace=True)

    sale_price = raw['LogSalePrice']
    raw = raw.drop(['LogSalePrice','Id'],axis=1)
    
    #Remove columns from train that are not in test data rather than making them 0 in test
    raw = raw.drop(['Exterior1st_ImStucc', 'Exterior1st_Stone', 'HouseStyle_2.5Fin'] ,axis=1)
    
    return(raw, sale_price)
    
def load_test():
    '''
    Reads in and cleans test data.
    Returns a data frame of cleaned test data
    '''
    
    raw_test = pd.read_csv(".//house-prices-advanced-regression-techniques/cleandata/s2_clean_dummified_test.csv")
    test_IDs = raw_test['Id']
    test_IDs = pd.DataFrame(test_IDs)
    raw_test.drop(['Id', 'Exterior1st_Other'], axis = 1, inplace = True)
    raw_test.drop(vif, axis=1, inplace=True)
    
    return(raw_test, test_IDs)


pd.set_option('display.max_columns', 150)
raw, sale_price = load_train()

vif = pd.read_csv('vif_scores.csv')
#Keep only those with score under ...
vif = vif.loc[vif['vif_score'] <2, 'variable']
vif  = list(vif)

raw = raw.drop(vif, axis=1)

X = raw.copy()
Y = sale_price.copy()

#get_ridge(X,Y, vif)
#get_lasso(X, Y, vif)
#get_linear(X, Y, vif)

get_stack(X, Y)
