import os
import datetime
# import quandl
import numpy as np 

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import pandas as pd 
from pandas_datareader import data as web

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style

import math

###########################################
# Get Netflix stock data form Yahoo Finance
###########################################

# get Netflix data from yahoo
df = web.DataReader('NFLX', 'yahoo', start="2019-01-01", end="today")
#print(df.tail())

# select 'Adj Close' column and calculate moving average for 100 days
close_prx = df['Adj Close']
# print(close_prx)
mavg = close_prx.rolling(100, center=True, min_periods=1).mean()
# print(mavg)

#df = df[['Adj Close']]
#print(df)

#################################################
# plot Netflix 'Adj Close' and 'Moving Avg' data
#################################################

mpl.rc('figure', figsize=(8,7))
style.use('ggplot')

close_prx.plot(label='NFLX')
mavg.plot(label='mavg')

plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('NFLX - Adj Close & Mov Avg')
plt.grid(True)
plt.legend()
plt.show()

###########################
#select features for model
###########################
dfreg = df.loc[:,['Adj Close', 'Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
#print(dfreg)

######################
# process data
######################

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)



#######################################################
# Forecast last 30 days from last reported 'Adj Close' 
########################################################
# Grab %1 of data for forecasting
forecast_out = int(math.ceil(1/8 * len(dfreg)))
# forecast_out = 30
forecast_col = 'Adj Close'
dfreg['Prediction'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['Prediction'], 1))

# Scale X
# X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(dfreg['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Using linear regression model
lr_model = LinearRegression(n_jobs=-1)
lr_model.fit(X_train, y_train)

# Using quadratic regression with 2 polynomial features
quad1_model = make_pipeline(PolynomialFeatures(2), Ridge())
quad1_model.fit(X_train, y_train)

quad2_model = make_pipeline(PolynomialFeatures(3), Ridge())
quad2_model.fit(X_train, y_train)

# Using SVM radias basis function (RBF) model
rbf_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
rbf_model.fit(X_train, y_train)

# KNN Regression
knn_model = KNeighborsRegressor(n_neighbors=2)
knn_model.fit(X_train, y_train)

# Get confidence scores for each model 
lr_confidence = lr_model.score(X_test, y_test)
quad1_confidence = quad1_model.score(X_test, y_test)
quad2_confidence = quad2_model.score(X_test, y_test)
rbf_confidence = rbf_model.score(X_test, y_test)
knn_confidence = knn_model.score(X_test, y_test)

# print confidence scores for each model--Quad 2 performs best
print("lr confidence: ", lr_confidence)
print("quad1 confidence with 2 poly features ", quad1_confidence)
print("quad2 confidence with 3 poly features ", quad2_confidence)
print("svm rbf confidence: ", rbf_confidence)
print("knn confidence: ", knn_confidence)

forecast_set = quad1_model.predict(X_lately)
dfreg['Forecast'] = np.nan

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

plt.title('NFLX - Adj CP 1/1/19 to 9/9/19 with 30-day forecast')

dfreg['Adj Close'].head(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()