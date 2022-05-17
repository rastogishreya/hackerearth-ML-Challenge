# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 02:09:58 2020

@author: rasto
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox as bx
from scipy.special import boxcox1p
import lightgbm as lgm
import xgboost as xgb

from catboost import CatBoostRegressor,Pool

train = pd.read_csv("Train.csv")
dem = train.iloc[:,6].values

"""finding corelation(heat Map)"""

cormat = train.corr()
plt.subplots(figsize = (12,9))
sns.heatmap(cormat,vmax = 0.9,vmin=0,square = True,annot = True)

""" Removing The outlers """
train.drop(train[(train['Demand']>30000)].index, inplace=True,axis=0)
train.drop(train[(train['High_Cap_Price']>50000)].index, inplace=True,axis=0)

"""taking low cap price from data """
low_cap = train.iloc[:,7].values
test = pd.read_csv("Test.csv")

dataset = pd.concat([train, test])
dataset.drop(["Item_Id"], axis = 1, inplace = True)
dataset.drop(["Low_Cap_Price"], axis = 1, inplace = True)
dataset.drop(["Date"], axis = 1, inplace = True)

fq = dataset.groupby('Grade').size()/len(dataset)    
# mapping values to dataframe 
dataset.loc[:, "{}_freq_encode".format('Grade')] = dataset['Grade'].map(fq)   
# drop original column. 
dataset = dataset.drop(['Grade'], axis = 1)  


"""Skewness"""
dataset['High_Cap_Price'] = bx(dataset['High_Cap_Price'], lmbda = 0.04 )
lmb = 0.005
dataset['Demand'] = boxcox1p(dataset['Demand'], lmb)
dataset['Grade_freq_encode'] = boxcox1p(dataset['Grade_freq_encode'], lmb)
dataset['Product_Category'] = boxcox1p(dataset['Product_Category'], lmb)
dataset.skew()

dtrain = dataset.head(9796)

dtest = dataset.tail(5763)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(dtrain, low_cap, test_size = 0.33, random_state = 0)

#using: Randomforest regressor by hyper parameter tuning
'''from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, n_jobs = -1)

regressor.fit(X_train, Y_train)

""" Using: LightGBM Regressor """

gbm = lgm.LGBMRegressor()
clf = gbm.fit(X_train,Y_train,
              eval_set = [(X_test,Y_test)],
              eval_metric ='l2',
              early_stopping_rounds = 500)


""" Using: Gradient Boosting Regressor"""

from sklearn.ensemble import   GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                  )
GBoost.fit(X_train,Y_train)

""" Using: CatBoost Regrerssor"""

hyperparam={'depth': 6,
            'l2_leaf_reg': 1,
            'iterations': 500,
            'learning_rate': 0.22,
            'grow_policy': 'SymmetricTree'}

from catboost import CatBoostRegressor
cat=CatBoostRegressor(**hyperparam,task_type='CPU')
cat.fit(X_train,Y_train)'''

""" Xgboost Regressor """
params = {'n_estimators': 100,
          'learning_rate': 500,
          'max_depth': 6,
          'booster': 'gbtree'
          }
from  xgboost import XGBRegressor
xg = xgb.XGBModel(**params)

xg.fit(X_train, Y_train,
        eval_set=[(X_test, Y_test)],
        eval_metric='logloss',
        )

Y_pred = xg.predict(X_test)

from sklearn.metrics import mean_squared_log_error
print("Score = ",max(0, (100 - mean_squared_log_error(Y_test,Y_pred))))



lol = (regressor.predict(dtest.values)+ GBoost.predict(dtest.values)+cat.predict(dtest.values) + classifier.predict(dtest.values))/ 4

ID=test["Item_Id"]
submissionRandom = pd.DataFrame(
    {'Item_Id': ID, 'Low_Cap_Price':Y_pred})
submissionRandom.to_csv('submissionww.csv', index=False)
