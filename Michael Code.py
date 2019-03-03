#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:12:47 2019

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
plt.style.use('ggplot')
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

os.chdir('/Users/michaelsankari/Documents/NYC Data Science/Machine Learning Project/github/ML-project-for-NYCDSA')
raw = pd.read_csv("./house-prices-advanced-regression-techniques/cleandata/s2_clean_dummified.csv")
raw.shape

pd.set_option('display.max_columns', 150)
raw.head()

sale_price = raw['LogSalePrice']
raw = raw.drop(['LogSalePrice','Id'],axis=1)

#Remove columns from train that are not in test data rather than making them 0 in test
raw = raw.drop(['Exterior1st_ImStucc', 'Exterior1st_Stone', 'HouseStyle_2.5Fin'] ,axis=1)

X = raw.copy()
Y = sale_price.copy()

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
ridge = Ridge()
lasso = Lasso()
net   = ElasticNet()

def ridge(X, Y):
    #running ridge with alpha 0 (MLR)
    ridge.set_params(alpha = 0, normalize = True)
    ridge.fit(X, Y)
    ridge.score(X, Y)
    
    #Grid search for ridge
    alphas_ridge = np.linspace(0,10,50)
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
    ridge_best.score(X,Y)

def lasso (X, Y):
    lasso.set_params(alpha = 0, normalize = True, max_iter = 10000)
    lasso.fit(X, Y)
    lasso.score(X, Y)
    
    alphas_lasso = np.logspace(-7, -2, 50)
    tuned_parameters_l = [{'alpha': alphas_lasso}]
    n_folds = 5
    cv = KFold(n_splits=n_folds, shuffle=True)
    
    tune_lasso = GridSearchCV(lasso, tuned_parameters_l, 
                              cv=cv, refit=True, return_train_score = True, 
                              scoring = 'neg_mean_squared_error')
    tune_lasso.fit(X,Y)
    
    print(tune_lasso.best_params_)
    #print(np.max(-tune_lasso.cv_results_['mean_test_score']))
    print(np.min(-tune_lasso.cv_results_['mean_test_score']))
    
    lasso_best = tune_lasso.best_estimator_
    lasso_best.fit(X,Y)
    lasso_best.score(X,Y)  
    
def export_to_file():
    raw_test = pd.read_csv(".//house-prices-advanced-regression-techniques/cleandata/s2_clean_dummified_test.csv")
    
    test_IDs = raw_test['Id']
    test_IDs = pd.DataFrame(test_IDs)
    raw_test.drop(['Id', 'Exterior1st_Other'], axis = 1, inplace = True)
    
    ## running lasso to predict housing values 
    predict_lasso = lasso_best.predict(raw_test)
    predict_lasso = np.exp(predict_lasso)
    predict_lasso = pd.DataFrame(predict_lasso)
    predict_lasso = pd.concat([test_IDs, predict_lasso], axis = 1)
    predict_lasso.columns = ['Id', 'SalePrice']
    predict_lasso.to_csv('remove cols in train not in test, remove outliers lasso.csv', index=False)
    
    ## running ridge to predict housing values 
    predict_ridge = ridge_best.predict(raw_test)
    predict_ridge = np.exp(predict_ridge)
    predict_ridge = pd.DataFrame(predict_ridge)
    predict_ridge = pd.concat([test_IDs, predict_ridge], axis = 1)
    predict_ridge.columns = ['Id', 'SalePrice']
    predict_ridge.to_csv('remove cols in train not in test ridge.csv', index=False)
