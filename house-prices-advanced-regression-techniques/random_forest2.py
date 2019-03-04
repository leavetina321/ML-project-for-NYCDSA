#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:11:27 2019

@author: yewanxin
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
le = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split
import pickle
import os

os.chdir('/Users/michaelsankari/Documents/NYC Data Science/Machine Learning Project/github/ML-project-for-NYCDSA/house-prices-advanced-regression-techniques')

train=pd.read_csv('./cleandata/traindata_full_dummified.csv')
test=pd.read_csv("./cleandata/testdata_full_dummified.csv")
### drop outliers
def find_outliers(df,col):
    mean = np.mean(df[col], axis=0)
    sd = np.std(df[col], axis=0)
    gap=mean-4*sd
    gap2=mean+4*sd
    outliers = df[ (df[col] > gap2) | (df[col]< gap)].index
    return outliers
miscval=list(find_outliers(train, 'MiscVal'))
lotarea=list(find_outliers(train, 'LotArea'))
totalbsmtsf=list(find_outliers(train,'TotalBsmtSF'))
grlivarea=list(find_outliers(train,'GrLivArea'))
droplist=miscval+lotarea+totalbsmtsf+grlivarea
train.drop(droplist,axis=0, inplace=True)

#
#X_train=pd.get_dummies(features, drop_first=True)
X_train=train.drop(['LogSalePrice','Id'],axis=1)
#x_test = pd.get_dummies(test, drop_first=True)
test_id=test['Id'].to_frame()
x_test=test.drop(['Id'],axis=1)  
y_train=train['LogSalePrice']

##predict test data
#randomForest.predict(test)
model=ensemble.RandomForestRegressor()
###Grid Search
grid_para_forest = [{
    "n_estimators": [650],
    "min_samples_leaf": [2],
    "min_samples_split": [2],
    "random_state": [42]}]
grid_search_forest = GridSearchCV(model, grid_para_forest, cv=5)
grid_search_forest.fit(X_train, y_train)

bestparam= grid_search_forest.best_params_
bestscore= grid_search_forest.best_score_

# Train test split
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_train, y_train,  random_state = 42, test_size = 0.2)


# Fit the model
model = ensemble.RandomForestRegressor(**bestparam)
#model.set_params(bestparam)
model.fit(X_train,y_train)
model.score(X_train,y_train)
model.fit(X_train_new, y_train_new)
model.score(X_train_new, y_train_new)
model.score(X_test_new, y_test_new)


##feature importance
feature_importance = list(zip(X_train.columns, model.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_importance = np.array(feature_importance, dtype=dtype)
feature_sort = np.sort(feature_importance, order='importance')[::-1]
name, score = zip(*list(feature_sort))
fea_i=pd.DataFrame({'name':name,'score':score})
fea_i[:10].plot.bar(x='name', y='score')
fea_i

my_fig = fea_i[:10].plot.bar(x='name', y='score')
my_fig.tick_params(labelsize=14)
my_fig.figure.savefig('my_fig.png', dpi=300, bbox_inches = 'tight')

### output predict model score 
predictscore=pd.DataFrame(model.predict(x_test),columns=['SalePrice'])
predictscore=pd.concat([test_id,predictscore],axis=1)
predictscore['SalePrice']=predictscore['SalePrice'].apply(lambda t: np.exp(t))
predictscore.to_csv('random_forest_score3.csv',index=False)

pickle.dump(model, open('random_forest.sav', 'wb'))
