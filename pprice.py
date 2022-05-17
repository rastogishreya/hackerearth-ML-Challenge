# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:51:01 2020

@author: rasto
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.special import boxcox1p
from catboost import CatBoostRegressor,Pool
Dtrain=pd.read_csv("Train.csv")
Dtest=pd.read_csv("Test.csv")

sns.heatmap(Dtrain.corr())


"""Understanding Outliers"""

fig, ax = plt.subplots()
ax.scatter(x = Dtrain['Demand'], y = Dtrain['Low_Cap_Price'])
plt.xlabel('Demand', fontsize=13)
plt.ylabel('Low_cap_price', fontsize=13)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x = Dtrain['State_of_Country'], y = Dtrain['Low_Cap_Price'])
plt.xlabel('State_of_Country', fontsize=13)
plt.ylabel('Low_cap_price', fontsize=13)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x = Dtrain['High_Cap_Price'], y = Dtrain['Low_Cap_Price'])
plt.xlabel('High_Cap_Price', fontsize=13)
plt.ylabel('Low_cap_price', fontsize=13)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x = Dtrain['Market_Category'], y = Dtrain['Low_Cap_Price'])
plt.xlabel('Market_Category', fontsize=13)
plt.ylabel('Low_cap_price', fontsize=13)
plt.show()

sns.lmplot(x='High_Cap_Price',y='Low_Cap_Price',data=Dtrain)





"""Understanding skewness """
(muH, sigmaH) = norm.fit(Dtrain['High_Cap_Price'])
sns.distplot(Dtrain['High_Cap_Price'] , fit=norm)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(muH, sigmaH)],
            loc='best')
plt.ylabel('Frequency')

(muL, sigmaL) = norm.fit(Dtrain['Low_Cap_Price'])
sns.distplot(Dtrain['Low_Cap_Price'] , fit=norm)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(muL, sigmaL)],
            loc='best')
plt.ylabel('Frequency')

(muH, sigmaH) = norm.fit(Dtrain['Grade'])
sns.distplot(Dtrain['Grade'] , fit=norm)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(muH, sigmaH)],
            loc='best')
plt.ylabel('Frequency')


Dtrain.drop(Dtrain[(Dtrain['Demand']>30000)].index, inplace=True,axis=0)
Dtrain.drop(Dtrain[(Dtrain['High_Cap_Price']>50000)].index, inplace=True,axis=0)







sns.heatmap(Dtrain.corr(), annot = True, vmin=-1, vmax=1, center= 0)


Df=pd.concat([Dtrain,Dtest])
Df=Df.drop(['Item_Id','Low_Cap_Price','Date'],axis=1)
Price=Dtrain['Low_Cap_Price']
ID=Dtest['Item_Id']

Df['High_Cap_Price']=boxcox1p(Df['High_Cap_Price'],0.04)
Df['Product_Category']=np.log1p(Df['Product_Category'])
lam=0.002
Df['Demand']= boxcox1p(Df['Demand'],lam)
Df['Grade']= boxcox1p(Df['Grade'],lam)
Df.skew()

Train=Df.head(9796)
Test=Df.tail(5763)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Train, Price, test_size=0.37)





from sklearn.ensemble import   GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                  )
GBoost.fit(X_train,y_train)






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



finalmeansol = (GBoost.predict(Test)+ cat.predict(Test.values) + model.predict(Test.values))/ 3
submissionRandom = pd.DataFrame(
    {'Item_Id': ID, 'Low_Cap_Price':finalmeansol})
submissionRandom.to_csv('submissionnew.csv', index=False)