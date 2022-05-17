""" Importing the important libraries """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox as bx
from scipy.special import boxcox1p

import lightgbm as lgm

train = pd.read_csv("Train.csv")


""" Data Preprocessing""""
                          

"""plotting the figures"""


fig, ax = plt.subplots()
ax.scatter(x = train['Product_Category'], y = train['Low_Cap_Price'])
plt.xlabel('Product_Category', fontsize = 13)
plt.ylabel('Low_Cap_Price', fontsize = 13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['Demand'], y = train['Low_Cap_Price'])
plt.xlabel('Demand', fontsize = 13)
plt.ylabel('Low_Cap_Price', fontsize = 13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['High_Cap_Price'], y = train['Low_Cap_Price'])
plt.xlabel('High_Cap_Price', fontsize = 13)
plt.ylabel('Low_Cap_Price', fontsize = 13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['Grade'], y = train['Low_Cap_Price'])
plt.xlabel('Grade', fontsize = 13)
plt.ylabel('Low_Cap_Price', fontsize = 13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['Market_Category'], y = train['Low_Cap_Price'])
plt.xlabel('Market_Category', fontsize = 13)
plt.ylabel('Low_Cap_Price', fontsize = 13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['Date'], y = train['Low_Cap_Price'])
plt.xlabel('Date', fontsize = 13)
plt.ylabel('Low_Cap_Price', fontsize = 13)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x = train['State_of_Country'], y = train['Low_Cap_Price'])
plt.xlabel('State_of_Country', fontsize = 13)
plt.ylabel('Low_Cap_Price', fontsize = 13)
plt.show()

sns.countplot(train['Demand'])
train.Demand.hist(bins = 10)
train.describe()
"""finding corelation(heat Map)"""

cormat = train.corr()
plt.subplots(figsize = (12,9))
sns.heatmap(cormat,vmax = 0.9,vmin=0,square = True,annot = True)



"""taking low cap price from data """
low_cap = train.iloc[:,7].values
test = pd.read_csv("Test.csv")

dataset = pd.concat([train, test])
dataset.drop(["Item_Id"], axis = 1, inplace = True)
dataset.drop(["Low_Cap_Price"], axis = 1, inplace = True)
dataset.drop(["Date"], axis = 1, inplace = True)

dataset.drop(dataset[(dataset['Demand']> 60000)].index, inplace = True,axis = 0)

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



""" Mean Encoding """

def Mean_encode():
    mean = Dtrain['Grade'].mean()

    agg = Dtrain.groupby('Grade')['Grade'].agg(['count','mean'])
    counts = agg['count']
    means = agg['mean']
    weight = 100

    smooth = ((counts * means) + (weight * mean)) / (counts + weight)

    Dtrain.loc[:,"{}_mean_encode".format('Grade')] = Dtrain['Grade'].map(smooth)
    Dtest.loc[:,"{}_mean_encode".format('Grade')] = Dtest['Grade'].map(smooth)

""" Removing the utliers in the dataset"""




'''sns.boxplot(x=dataset['Demand'])
sns.boxplot(x=dataset['Grade'])
sns.boxplot(x=dataset['Market_Category'])
sns.boxplot(x=dataset['State_of_Country'])
sns.boxplot(x=dataset['High_Cap_Price'])
sns.boxplot(x=dataset['Product_Category'])'''



dtrain = dataset.head(9798)

dtest = dataset.tail(5763)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(dtrain, low_cap, test_size = 0.33, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
dtest = sc.transform(dtest)

#using Randomforest regressor by hyper parameter tuning
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1200, n_jobs = -1)
regressor.fit(X_train, Y_train)



#Grid Search
'''collection = {'n_estimators': [200,300,500],
              'max_features': ["auto","sqrt","log2"], 
              'max_depth':[10,20,30,'None']}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(regressor, collection, cv = 3)
random_search.fit(X_train, Y_train)
random_search.best_params_

Y_pred = random_search.predict(X_test)'''


#Using LightGBM library
gbm = lgm.LGBMRegressor()
clf = gbm.fit(X_train,Y_train,
              eval_set = [(X_test,Y_test)],
              eval_metric ='l2',
              early_stopping_rounds = 500)


params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mse'
params['sub_feature'] = 0.5
params['num_leaves'] = 100
params['min_data'] = 70
params['max_depth'] = 20

#Grid Search
collection = {'n_estimators': [200,300,500],
              'learning_rate': [0.001,0.005,0.03], 
              'num_leaves': [50,100,150],
              'max_bin': [300,400,500],
              'boosting_type': ['gbdt','goss','dart'],
              }

from sklearn.model_selection import GridSearchCV
gridsearch = GridSearchCV(estimator = gbm,param_grid = collection, cv = 3  )
gridsearch.fit(X_train, Y_train,
               eval_set = [(X_test,Y_test)],
               eval_metric ='l2',
               early_stopping_rounds = 500)
gridsearch.best_params_

""" Gradient Boosting Regressor"""

from sklearn.ensemble import   GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                  )
GBoost.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)

hyperparam={'depth': 6,
            'l2_leaf_reg': 1,
            'iterations': 500,
            'learning_rate': 0.22,
            'grow_policy': 'SymmetricTree'}

from catboost import CatBoostRegressor
cat=CatBoostRegressor(**hyperparam,task_type='CPU')
cat.fit(X_train,Y_train)

'''from sklearn.metrics import mean_squared_log_error
print("Score = ",100 - mean_squared_log_error(Y_test,Y_pred))'''

#using knn
'''from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import mean_squared_log_error
print("Score = ",100 - mean_squared_log_error(Y_test,Y_pred))'''


lol = (clf.predict(dtest.values)+ regressor.predict(dtest.values)+ GBoost.predict(dtest)+cat.predict(dtest.values))/ 4

from sklearn.metrics import mean_squared_log_error
print("Score = ",max(0, (100 - mean_squared_log_error(Y_test,Y_pred))))

ID=test["Item_Id"]
submissionRandom = pd.DataFrame(
    {'Item_Id': ID, 'Low_Cap_Price':lol})
submissionRandom.to_csv('submission.csv', index=False)




