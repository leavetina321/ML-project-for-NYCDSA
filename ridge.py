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

def get_ridge(X, Y):
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
    model_filename = 'ridge' + suffix +'.sav'
    pickle.dump(ridge_best, open(model_filename, 'wb'))
    csv_filename = 'ridge ' + suffix + '.csv'
    
    raw_test, test_IDs = load_test()
    predict = ridge_best.predict(raw_test)
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
    os.chdir('/Volumes/michaelsankari/Documents/NYC Data Science/Machine Learning Project/github/ML-project-for-NYCDSA')
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
    
    raw_test = pd.read_csv("./house-prices-advanced-regression-techniques/cleandata/s2_clean_dummified_test.csv")
    test_IDs = raw_test['Id']
    test_IDs = pd.DataFrame(test_IDs)
    raw_test.drop(['Id', 'Exterior1st_Other'], axis = 1, inplace = True)
    raw_test.drop(vif, axis=1, inplace=True)
    
    return(raw_test, test_IDs)


pd.set_option('display.max_columns', 150)
raw, sale_price = load_train()

X = raw.copy()
Y = sale_price.copy()

data = list(zip(X.columns, ridge_best.coef_))
data = pd.DataFrame(data, columns = ['Name', 'Coef'])
data.sort_values('Coef')

