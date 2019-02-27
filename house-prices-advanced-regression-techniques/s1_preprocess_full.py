# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:30:40 2019

@author: yanqi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:01:52 2019

@author: yanqi
"""

# This file works on the raw input data and perform the following
# 1. Imputation: (see impvars)
#    - convert NA to "None" where makes sense
#    - customized imputation for some vars 
# 2. Modification: combine categories etc (see modvars)
# 3. Create new features: (see newvars)
# 4. Drop variables: (see dropvars)
# 5. Creates a few output csv files for analyses
#   - clean_fulldata.csv: output after steps 1-3
#   - clean_reducedata.csv: output after steps 1-4

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy  as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from sklearn import linear_model
plt.style.use('ggplot')

def chk_mv(df):
    mv_bycol = pd.DataFrame( df.isnull().sum(axis=0), columns = ['num_mv'])
    mv_bycol['pct_mv'] = mv_bycol['num_mv']/df.shape[0]
    mv_bycol = mv_bycol.sort_values('num_mv', ascending=False)
    mv_by_col = mv_bycol[mv_bycol['num_mv'] > 0]
    print(mv_by_col)
    
def chk_LotFrontage_nhood(df1): 
    sns.lmplot(y = 'LotFrontage', x = 'LotArea', data = df1[df1.LotArea < 20000], col = 'Neighborhood', \
           sharey = False, sharex = False, height = 3, col_wrap = 3, scatter_kws={'s':8, 'alpha':0.5, 'edgecolor':"black"})   # 

def chk_porch(df1):
    df = df1.copy()
    df['Porch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['Porch2'] = df['OpenPorchSF'] + df['ScreenPorch']
    df['PorDeck'] = df['OpenPorchSF'] + df['WoodDeckSF']
    df['PorDeck2'] = df['OpenPorchSF'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['PorDeck3'] = df['Porch'] + df['WoodDeckSF']
    
    r = df[ ['WoodDeckSF','PorDeck','PorDeck2','PorDeck3','Porch','Porch2',\
         'OpenPorchSF','EnclosedPorch', '3SsnPorch','ScreenPorch','SalePrice'] ].corr() 

    r1 = r[['SalePrice']]

    print(r)
    print(r1)

def fill_mv_LotFrontage(df,r0):
    ols = linear_model.LinearRegression()
    nhoods = df.Neighborhood.unique()

    for nhood in nhoods:
        df_n = df[ df['Neighborhood'] == nhood ]
        mv_idx = (df['Neighborhood'] == nhood) & (df['LotFrontage'].isnull())
        
        # if there are mv in this neighborhood
        if np.sum(mv_idx) > 0:
            X = np.array(df_n.loc[ df_n['LotFrontage'].notnull(),['LotArea']]).reshape(-1,1)
            Y = np.array(df_n.loc[ df_n['LotFrontage'].notnull(), ['LotFrontage']])
            ols.fit(X,Y)
            R2 = ols.score(X,Y)
            print(nhood, "R^2: %.2f" %R2, "beta_1: %.3f" %ols.coef_, "beta_0: %.3f" %ols.intercept_)
        
            # if neighborhood based regression on LotArea has decent R^2
            if R2 > r0:
                df.loc[ mv_idx , ['LotFrontage'] ] = ols.predict( np.array(df.loc[mv_idx, 'LotArea' ]).reshape(-1,1) )
                print("imputed with regression \n", df.loc[ mv_idx , ['LotFrontage'] ],"\n" )
            else:
                df.loc[ mv_idx , ['LotFrontage'] ] = np.median(Y)
                print("imputed with neighborhood median \n",  df.loc[ mv_idx , ['LotFrontage'] ],"\n" )
    return df

proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Python Machine Learning\\Housing Price Prediction\\house_price_prediction\\code'
os.chdir(proj_path)

raw = pd.read_csv('../data/train.csv')
#raw = raw.drop('Id',axis=1)
raw.shape
pd.set_option('display.max_columns', 90)
raw.head()

# create dictionaries to keep track of changes to variables
newvars = []  # new variables created
impvars = dict()  # variables imputed
modvars = dict()  # other changes to variables, such as combining categories
dropvars = dict()

chk_mv(raw)

# Missing Value - Most common case: NA means None, e.g. NA for Alley means "No access to Alley"
df1 = raw.copy()
NA2None = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1', \
           'BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt','GarageFinish', \
           'GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

for name in NA2None:
    df1.loc[ df1[name].isnull(), [name] ] = 'None'
    impvars[name] = "NA changed to None"

# Missing value - special cases
# MasVnrArea, Electrical
df1.loc[ df1['MasVnrArea'].isnull(), ['MasVnrArea'] ] = 0
impvars['MasVnrArea'] = "NA changed to 0"

df1.loc[ df1['Electrical'].isnull(), ['Electrical','YearBuilt'] ]
df1.loc[ df1['YearBuilt'] == 2006 ].Electrical.value_counts()
df1.loc[ df1['Electrical'].isnull(), ['Electrical'] ] = 'SBrkr'  # all houses built in 2006 has SBrkr 
impvars['Electrical'] = "NA changed to SBrkr"

# From scatterplot of LotFrontage vs. LotArea, approximately linear relation 
df1 = fill_mv_LotFrontage(df1,0.5)
impvars['LotFrontage'] = "Imputed based on within neighborhood regression on LotArea if R^2>0.5, otherwise using neighborhood median LotFrontage"
chk_mv(df1)

# Functional: change to two categories (Typ or NonTyp)
df1['Functional'].value_counts()
df1.loc[ df1['Functional'] != 'Typ', ['Functional'] ] = 'NonTyp'
df1['Functional'].value_counts()
modvars['Functional'] = "changed to 2 categories Typ & NonType"

# PavedDrive: combine P and N
df1['PavedDrive'].value_counts()
df1.loc[ df1['PavedDrive'] == 'P', ['PavedDrive'] ] = 'N'
df1['PavedDrive'].value_counts()
modvars['PavedDrive'] = "combined P & N"

# PoolQC: convert to binary 0 = No pool, 1 = has pool
print(df1['PoolQC'].value_counts())
idx_none = (df1['PoolQC'] == 'None')
idx_other = (df1['PoolQC'] != 'None')
df1.loc[ idx_none , ['PoolQC'] ] = 0
df1.loc[ idx_other, ['PoolQC'] ] = 1
print(df1['PoolQC'].value_counts())
modvars['PoolQC'] = "converte to 0 = No pool, 1 = has pool"

# Create 2 new house age Variables from YearBuilt, YearRemodAdd, YrSold
df1['Age'] = df1['YrSold'] - df1['YearBuilt']
df1['Re_Age'] = df1['YrSold'] - df1['YearRemodAdd']

# MoSold: create SeasonSold (Spring, Summer, Fall, Winter), later to drop MoSold
df1['MoSold'] = df1['MoSold'].apply(str)


# Create new variables from Condition1 and Condition2
print(df1['Condition1'].value_counts())
print(df1['Condition2'].value_counts())
df1.groupby(['Condition1','Condition2'])['Condition1'].agg(['count'])
allcon = np.union1d(df1['Condition1'].unique(),df1['Condition2'].unique())

for con in allcon:
    df1[con] = 0
    df1.loc[ (df1['Condition1'] == con) | (df1['Condition2'] == con) , [con] ] = 1
    newvars.append(con)
    print(df1[con].value_counts(),"\n")
    
# log transform response variable to make it normally distributed
df1['LogSalePrice'] = np.log(df1['SalePrice'])


# remove 2 outliers 
outliers = df1[ (df1['GrLivArea'] > 4000) & (df1['LogSalePrice'] < 13) ].index
df1.drop(outliers,axis=0, inplace=True)


df1.drop(['YrSold','YearBuilt','YearRemodAdd','Norm','Condition1', \
          'Condition2', 'GarageYrBlt', 'SalePrice'], axis = 1,inplace=True)
    
# these include 8 columns from Condition1 and Condition2, already 0/1 valued from preprocessing 
num_cols = df1._get_numeric_data().columns  
cat_cols = list(set(df1.columns) - set(num_cols))
df1_dummy = pd.get_dummies(df1[cat_cols], drop_first= True)
df1.drop(cat_cols,axis=1,inplace=True)
df1 = pd.concat([df1,df1_dummy],axis=1)

# write this dataset to csv
outfname1 = "../results/fulldatav2_dummified.csv"
df1.to_csv(outfname1, index = False)

print(df1.shape[1])

