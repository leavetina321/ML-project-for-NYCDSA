#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:59:30 2019

@author: michaelsankari
"""
import os
from mlxtend.regressor import StackingRegressor
from mlxtend.data import boston_housing_data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import pickle

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
    
    return(raw_test, test_IDs)

raw, sale_price = load_train()

X = raw.copy()
Y = sale_price.copy()

lr = LinearRegression()
ridge = Ridge(random_state=1)
svr_rbf = SVR(kernel='rbf')

stregr = StackingRegressor(regressors=[lr, ridge], 
                           meta_regressor=svr_rbf)

# Training the stacking classifier

stregr.fit(X, Y)
stregr.predict(X)

# Evaluate and visualize the fit

print("Mean Squared Error: %.4f"
      % np.mean((stregr.predict(X) - Y) ** 2))
print('Variance Score: %.4f' % stregr.score(X, Y))

pickle.dump(stregr, open('stack-3-1-19.sav', 'wb'))

raw_test, test_IDs = load_test()
predict = stregr.predict(raw_test)
predict = np.exp(predict)
predict = pd.DataFrame(predict)
predict = pd.concat([test_IDs, predict], axis = 1)
predict.columns = ['Id', 'SalePrice']
predict.to_csv('stack-3-1-19.csv', index=False)