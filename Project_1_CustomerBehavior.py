'''
File              : Project_1_CustomerBehavior.py
Name              : Senthilraj Srirangan
Exercise Details  :
    Build Customer Behavior Model.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Reading shopping data

X_train = pd.read_csv('online_shoppers_intention.csv')
df = X_train.copy()
print(df.head())

# Producing dummy variables for categorical data and cleaning data

dummiesdf = pd.get_dummies(df['VisitorType'])
df.drop('VisitorType', inplace = True, axis = 1)
df['New_Visitor'] = dummiesdf['New_Visitor']
df['Other'] = dummiesdf['Other']
df['Returning_Visitor'] = dummiesdf['Returning_Visitor']

dfmonth = pd.get_dummies(df['Month'])
df.drop('Month', inplace = True, axis = 1)
dfwithdummies = pd.concat([df, dfmonth], axis = 1, sort = False)

dfwithdummies['Class'] = df['Revenue'].astype(int)
dfwithdummies.drop('Revenue', axis = 1, inplace = True)
dfwithdummies['Weekend'] = df['Weekend'].astype(int)
dfwithdummies.drop('Returning_Visitor', axis = 1, inplace = True)
dfcleaned = dfwithdummies.copy()


X = dfcleaned.drop('Class', axis = 1)
Y = dfcleaned['Class'].copy()

# Checking for Collinearity Between Features and Creating Reducing Feature Size

cor = X.corr()

sns.heatmap(cor, xticklabels=cor.columns,yticklabels=cor.columns)
plt.show()

# Quick overview of features

# Histogram of all features
for idx,column in enumerate(X.columns):
    plt.figure(idx)
    X.hist(column=column,grid=False)
    plt.show()

# Checking for NA values
for i in X.columns:
    print('Feature:',i)
    print('# of N/A:',X[i].isna().sum())


for i in X_train.columns:
    print('####################')
    print('COLUMN TITLE:',i)
    print('# UNIQUE VALUES:',len(X_train[i].unique()))
    print('UNIQUE VALUES:',X_train[i].unique())
    print('####################')
    print()


# Scaling to normalize data
X_copy = X.copy()
rc = RobustScaler()
X_rc=rc.fit_transform(X_copy)
X_rc=pd.DataFrame(X_rc,columns=X.columns)

for idx,column in enumerate(X_rc.columns):
    plt.figure(idx)
    X_rc.hist(column=column,grid=False)


# Linear Model with All Features

from sklearn import linear_model
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X_rc,Y,test_size=.2)

# Linear model
model = linear_model.SGDClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print('Linear Model Accuracy : ',accuracy_score(y_test, y_pred))

from sklearn.metrics import roc_auc_score
print('Linear Model AUC Score : ',roc_auc_score(y_test, y_pred))

# Random Forest with all Features¶

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=17, random_state=0)
clf.fit(X_train, y_train)
y_pred1 = clf.predict(X_test)
print('Random Forest Accuracy',accuracy_score(y_test, y_pred1))
print('Random Forest AUC Score',roc_auc_score(y_test, y_pred1))
