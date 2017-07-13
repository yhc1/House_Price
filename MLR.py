#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

# pandas, numpy
import pandas as pd
import numpy as np

# statsmodels
import statsmodels.formula.api as sm

# machine learning
from sklearn.linear_model import LinearRegression

Data_train = pd.read_csv('data/train_numer.csv')
numeric = [col for col in Data_train.columns if Data_train.dtypes[col] != 'object']
train = pd.read_csv('data/train_numer.csv', usecols=numeric)
X_train = train.iloc[:, 2:len(train.columns)-1].values
y_train = train.iloc[:, len(train.columns)-1].values

numeric.remove('SalePrice')
test = pd.read_csv('data/test_numer.csv', usecols=numeric)
X_test = test.iloc[:, 2:len(train.columns)-1].values

MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

X_train = np.delete(X_train, 28, axis=1)
X_test = np.delete(X_test, 28, axis=1)
MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

X_train = np.delete(X_train, 28, axis=1)
X_test = np.delete(X_test, 28, axis=1)
MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

X_train = np.delete(X_train, 14, axis=1)
X_test = np.delete(X_test, 14, axis=1)
MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

X_train = np.delete(X_train, 25, axis=1)
X_test = np.delete(X_test, 25, axis=1)
MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

X_train = np.delete(X_train, 21, axis=1)
X_test = np.delete(X_test, 21, axis=1)
MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

X_train = np.delete(X_train, 24, axis=1)
X_test = np.delete(X_test, 24, axis=1)
MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

X_train = np.delete(X_train, 22, axis=1)
X_test = np.delete(X_test, 22, axis=1)
MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

X_train = np.delete(X_train, 11, axis=1)
X_test = np.delete(X_test, 11, axis=1)
MLR_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
MLR_OLS.summary()

y_pred = MLR_OLS.predict(X_test)


submission = pd.DataFrame({
            'Id':test['Id'],
            'SalePrice':y_pred
        })
submission.to_csv('Submission2.csv', index=False)   