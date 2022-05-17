# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 19:22:40 2020

@author: rasto
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.special import boxcox1p,boxcox
from catboost import CatBoostRegressor

Dtrain=pd.read_csv("Train.csv")
Dtest=pd.read_csv("Test.csv")


Dtrain.drop(Dtrain[(Dtrain['Demand']>30000)].index, inplace=True,axis=0)
Dtrain.drop(Dtrain[(Dtrain['High_Cap_Price']>50000)].index, inplace=True,axis=0)
                                                                  


Df=pd.concat([Dtrain,Dtest])
Df['Date'] = Df['Date'].astype(str)

def freq_encode(col):
    fe = Df.groupby(col).size()/len(Df)
    Df.loc[:, "{}_freq_encode".format(col)] = Df[col].map(fe) 

#freq_encode('Grade')
freq_encode('Date')
#freq_encode('Product_Category')
Df=Df.drop(['Item_Id','Low_Cap_Price','Grade','Date'],axis=1)
Price=Dtrain['Low_Cap_Price']
ID=Dtest['Item_Id']



sns.heatmap(Df.corr(), annot = True, vmin=-1, vmax=1, center= 0)


Df['High_Cap_Price']=boxcox(Df['High_Cap_Price'],0.04)
Df['Product_Category']=boxcox(Df['Product_Category'],0.002)
Df['Date_freq_encode']=boxcox(Df['Date_freq_encode'],0.23)
Df['State_of_Country']=boxcox(Df['State_of_Country'],7)
Df['Demand']= boxcox(Df['Demand'],0.13)



  

#Df['Demand']= boxcox(Df['Demand'],0.15)
#Df['Grade_freq_encode']= boxcox1p(Df['Grade_freq_encode'],0.003)
Df.skew()

Train=Df.head(9796)
Test=Df.tail(5763)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Train, Price, test_size=0.33)





from sklearn.ensemble import   GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                  )

GBoost.fit(X_train,y_train)

from lightgbm import LGBMRegressor
lightgbm = LGBMRegressor(objective='regression').fit(X_train,y_train)




from sklearn.ensemble import  RandomForestRegressor
model = RandomForestRegressor(n_estimators=500,n_jobs=-1)
model.fit(X_train,y_train)

x_test_result=(model.predict(X_test))
score=(max(0,(100-msle(y_test,x_test_result))))

y_pred=model.predict(Test)

hyperparam={'depth': 6,
            'l2_leaf_reg': 1,
            'iterations': 500,
            'learning_rate': 0.22,
            'grow_policy': 'SymmetricTree'}

from catboost import CatBoostRegressor
cat=CatBoostRegressor(**hyperparam,task_type='CPU')
cat.fit(X_train,y_train)



finalmeansol = (GBoost.predict(Test)+ cat.predict(Test.values)+model.predict(Test.values) +lightgbm.predict(Test.values))/ 4
submissionRandom = pd.DataFrame(
    {'Item_Id': ID, 'Low_Cap_Price':finalmeansol})
submissionRandom.to_csv('submissionneww_one.csv', index=False)



