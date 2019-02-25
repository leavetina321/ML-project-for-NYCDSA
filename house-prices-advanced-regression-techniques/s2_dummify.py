# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:17:00 2019

@author: yanqi
"""

import os
import pandas as pd

proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Python Machine Learning\\Housing Price Prediction\\house_price_prediction\\code'
os.chdir(proj_path)

infname = "../results/s1_clean_reducedata.csv"
df = pd.read_csv(infname)

# Figure out which columns need to be dummified
cols = df.columns

# these include 8 columns from Condition1 and Condition2, already 0/1 valued from preprocessing 
num_cols = df._get_numeric_data().columns  
cat_cols = list(set(cols) - set(num_cols))

# calculate total number of dummy variables 
tot_dummy = 0
cnt_dummy = {}
for v in cat_cols:
    cnt_dummy[v] = len(df[v].unique()) - 1
    tot_dummy +=  cnt_dummy[v]
print(tot_dummy)
print(cnt_dummy)

# now dummify each of those variables, drop the original categorical, merge dummies into dataframe
df_dummy = pd.get_dummies(df[cat_cols], drop_first= True)
df.drop(cat_cols,axis=1,inplace=True)

df = pd.concat([df,df_dummy],axis=1)

outfname = "../results/s2_clean_dummified.csv"
df.to_csv(outfname, index = False)