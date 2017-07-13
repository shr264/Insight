#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:42:31 2017

@author: syedrahman
"""
#########################################################################################
# This file contains all the helper functions for this project. Most functions take in  #
# an vector/matrix of responses (y) and a matrix of exegenous variables(X) and          #
# return a vector/matrix of predicted value over the test set, as well as the mse.      #
#                                                                                       #
# The methods implemented are an L1 penalised VAR, a RNN with an LSTM layer, facebook's #
# prophet and a Bayesian Lasso with the horseshoe prior. The training set is            #
# 2013-10-01 to 2016-04-01, while the testing set it 2016-05-01 to 2017-04-01.          #
# In some cases, as in the L1-VAR we had to do three splits for tuning parameter        #
# selection. L1-VAR estimates all neighborhoods at the same time while all the other    #
# do one at a time.                                                                     #
#########################################################################################

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.linear_model import Lasso, LassoCV,  MultiTaskLassoCV, MultiTaskLasso
from pandas.tools.plotting import autocorrelation_plot
import networkx as nx
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def findLassoPreds(alpha,y,X, returnPred = True):
    X_train, X_test = X.loc['2013-10-01':'2016-04-01'], X.loc['2016-05-01':'2017-04-01']
    y_train, y_test = y.loc['2013-10-01':'2016-04-01'], y.loc['2016-05-01':'2017-04-01']
    datestotest = y_test.index
    dt = datestotest[0] 
    lassoreg2 = MultiTaskLasso(alpha = alpha,max_iter=1e5)
    lassoreg2.fit(X_train,y_train)
    y_pred2 = lassoreg2.predict(X_test.loc[dt].reshape(1,-1))
    y_pred2 = pd.DataFrame(y_pred2) 
    y_pred2.columns = y.columns
    prediction = y_pred2
    X_train = X.loc['2013-10-01':dt]
    y_train = y.loc['2013-10-01':dt]
    for dt in datestotest[1:]:
        lassoreg2 = MultiTaskLasso(alpha = alpha,max_iter=1e5)
        lassoreg2.fit(X_train,y_train)
        y_pred2 = lassoreg2.predict(X_test.loc[dt].reshape(1,-1))
        y_pred2 = pd.DataFrame(y_pred2) 
        y_pred2.columns = y.columns
        prediction = pd.concat([prediction,y_pred2]) 
        X_train = X.loc['2013-10-01':dt]
        y_train = y.loc['2013-10-01':dt]
    prediction.index = y_test.index
    return ((y_test,prediction),mean_squared_error(y_test, prediction))
    
def findLassoAlpha(alpha,y,X, returnPred = False):
    X_train, X_test = X.loc['2013-10-01':'2015-04-01'], X.loc['2015-05-01':'2016-04-01']
    y_train, y_test = y.loc['2013-10-01':'2015-04-01'], y.loc['2015-05-01':'2016-04-01']
    datestotest = y_test.index
    dt = datestotest[0] 
    lassoreg2 = MultiTaskLasso(alpha = alpha,max_iter=1e5)
    lassoreg2.fit(X_train,y_train)
    y_pred2 = lassoreg2.predict(X_test.loc[dt].reshape(1,-1))
    y_pred2 = pd.DataFrame(y_pred2)
    y_pred2.columns = y.columns
    prediction = y_pred2
    X_train = X.loc['2013-10-01':dt]
    y_train = y.loc['2013-10-01':dt]
    for dt in datestotest[1:]:
        lassoreg2 = MultiTaskLasso(alpha = alpha,max_iter=1e5)
        lassoreg2.fit(X_train,y_train)
        y_pred2 = lassoreg2.predict(X_test.loc[dt].reshape(1,-1))
        y_pred2 = pd.DataFrame(y_pred2)
        y_pred2.columns = y.columns
        prediction = pd.concat([prediction,y_pred2]) 
        X_train = X.loc['2013-10-01':dt]
        y_train = y.loc['2013-10-01':dt]
    prediction.index = y_test.index
    if(returnPred):
        return (y_test,prediction)
    else:
        return mean_squared_error(y_test, prediction) 

def findProphetPreds(y, returnPred = True, nbhd = 'Gramercy Park and Murray Hill', 
                     categ = 'rents'):
    y = y[categ+nbhd]
    y_train, y_test = y.loc['2013-10-01':'2016-04-01'], y.loc['2016-05-01':'2017-04-01']
    fby_train = pd.DataFrame(y_train)
    datestotest = y_test.index
    dt = datestotest[0] 
    fby_train['ds'] = fby_train.index
    fby_train.columns =  ['y', 'ds']
    fby_train.index = range(len(fby_train))
    m = Prophet()
    m.fit(fby_train)
    future = pd.DataFrame(y.index[y.index<=dt])
    future.columns = ['ds']
    y_pred2 = m.predict(future)
    prediction = y_pred2[y_pred2['ds']==dt][['ds', 'yhat']]
    y_train = y.loc['2013-10-01':dt]
    for dt in datestotest[1:]:
        fby_train = pd.DataFrame(y_train)
        fby_train['ds'] = fby_train.index
        fby_train.columns =  ['y', 'ds']
        fby_train.index = range(len(fby_train))
        m = Prophet()
        m.fit(fby_train)
        future = pd.DataFrame(y.index[y.index<=dt])
        future.columns = ['ds']
        y_pred2 = m.predict(future)
        y_pred2 = y_pred2[y_pred2['ds']==dt][['ds', 'yhat']]
        prediction = pd.concat([prediction,y_pred2]) 
        y_train = y.loc['2013-10-01':dt]
    prediction.index = prediction.ds
    prediction = prediction.drop('ds', axis = 1)
    return ((y_test,prediction),mean_squared_error(y_test, prediction))

def findRNNPreds(y,X, returnPred = True, nbhd='Gramercy Park and Murray Hill', 
                 categ = 'rents'):
    np.random.seed(7)
    X_train, X_test = X.loc['2013-10-01':'2016-04-01'], X.loc['2016-05-01':'2017-04-01']
    y_train, y_test = y.loc['2013-10-01':'2016-04-01'], y.loc['2016-05-01':'2017-04-01']
    datestotest = y_test.index
    dt = datestotest[0] 
    X_trainRNN = X_train.as_matrix()
    X_trainRNN = X_trainRNN.reshape(X_trainRNN.shape[0], 1, X_trainRNN.shape[1])
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(y_train[categ+nbhd])
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_trainRNN.shape[1], X_trainRNN.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_trainRNN, dataset, epochs=100, batch_size=1, verbose=2)
    datestotest = y_test.index
    X_to_test = X_test.loc[dt].reshape(1,1,X_test.loc[dt].shape[0])
    yhat = model.predict(X_to_test, batch_size=1)
    invertedyhat = scaler.inverse_transform(yhat)
    prediction = invertedyhat
    X_train = X.loc['2013-10-01':dt]
    y_train = y.loc['2013-10-01':dt]
    for dt in datestotest[1:]:
        print 'time:',dt,'\n'  
        X_trainRNN = X_train.as_matrix()
        X_trainRNN = X_trainRNN.reshape(X_trainRNN.shape[0], 1, X_trainRNN.shape[1])
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(y_train[categ+nbhd])
        model = Sequential()
        model.add(LSTM(4, batch_input_shape=(1, X_trainRNN.shape[1], X_trainRNN.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_trainRNN, dataset, epochs=100, batch_size=1, verbose=2)
        datestotest = y_test.index
        X_to_test = X_test.loc[dt].reshape(1,1,X_test.loc[dt].shape[0])
        yhat = model.predict(X_to_test, batch_size=1)
        invertedyhat = scaler.inverse_transform(yhat)
        prediction = np.append(prediction,invertedyhat)
        X_train = X.loc['2013-10-01':dt]
        y_train = y.loc['2013-10-01':dt]
    prediction = pd.Series(prediction)
    prediction.index = y_test.index
    return ((y_test[categ+nbhd],prediction) , 
     mean_squared_error(y_test[categ+nbhd],prediction))
    
def findLassoPredsFcst(alpha,y,X, returnPred = True):
    X_train, X_test = X.loc['2013-10-01':'2017-04-01'], X.loc['2017-05-01':'2018-01-01']
    y_train, y_test = y.loc['2013-10-01':'2017-04-01'], y.loc['2017-05-01':'2018-04-01']
    datestotest = y_test.index
    dt = datestotest[0] 
    lassoreg2 = MultiTaskLasso(alpha = alpha,max_iter=1e5)
    lassoreg2.fit(X_train,y_train)
    y_pred2 = lassoreg2.predict(X_test.loc[dt].reshape(1,-1))
    y_pred2 = pd.DataFrame(y_pred2) 
    y_pred2.columns = y.columns
    prediction = y_pred2
    X_train = X.loc['2013-10-01':dt]
    y_train = y.loc['2013-10-01':dt]
    for dt in datestotest[1:]:
        lassoreg2 = MultiTaskLasso(alpha = 200,max_iter=1e5)
        lassoreg2.fit(X_train,y_train)
        y_pred2 = lassoreg2.predict(X_test.loc[dt].reshape(1,-1))
        y_pred2 = pd.DataFrame(y_pred2) 
        y_pred2.columns = y.columns
        prediction = pd.concat([prediction,y_pred2]) 
        X_train = X.loc['2013-10-01':dt]
        y_train = y.loc['2013-10-01':dt]
    prediction.index = y_test.index
    return ((y_test,prediction),mean_squared_error(y_test, prediction))
    
def findLassoAlphaFcst(alpha,y,X, returnPred = False):
    X_train, X_test = X.loc['2013-10-01':'2016-04-01'], X.loc['2016-05-01':'2017-04-01']
    y_train, y_test = y.loc['2013-10-01':'2016-04-01'], y.loc['2016-05-01':'2017-04-01']
    datestotest = y_test.index
    dt = datestotest[0] 
    lassoreg2 = MultiTaskLasso(alpha = alpha,max_iter=1e5)
    lassoreg2.fit(X_train,y_train)
    y_pred2 = lassoreg2.predict(X_test.loc[dt].reshape(1,-1))
    y_pred2 = pd.DataFrame(y_pred2)
    y_pred2.columns = y.columns
    prediction = y_pred2
    X_train = X.loc['2013-10-01':dt]
    y_train = y.loc['2013-10-01':dt]
    for dt in datestotest[1:]:
        lassoreg2 = MultiTaskLasso(alpha = 200,max_iter=1e5)
        lassoreg2.fit(X_train,y_train)
        y_pred2 = lassoreg2.predict(X_test.loc[dt].reshape(1,-1))
        y_pred2 = pd.DataFrame(y_pred2)
        y_pred2.columns = y.columns
        prediction = pd.concat([prediction,y_pred2]) 
        X_train = X.loc['2013-10-01':dt]
        y_train = y.loc['2013-10-01':dt]
    prediction.index = y_test.index
    if(returnPred):
        return (y_test,prediction)
    else:
        return mean_squared_error(y_test, prediction)

def findBayesHSPred(y,X,categ = 'rents',
                    nbhd='Gramercy Park and Murray Hill'):
    y = y[categ+nbhd]
    X_train, X_test = X.loc['2013-10-01':'2016-04-01'], X.loc['2016-05-01':'2017-04-01']
    y_train, y_test = y.loc['2013-10-01':'2016-04-01'], y.loc['2016-05-01':'2017-04-01']
    datestotest = y_test.index
    dt = datestotest[0]
    n, p = X_train.shape
    data = dict(n=n, p=p, X=X_train, y=y_train)
    model_code = """
        data {
        int<lower=0> n;
        int<lower=0> p;
        matrix[n,p] X;
        vector[n] y;
        }
        parameters {
        vector[p] beta;
        vector<lower=0>[p] lambda;
        real<lower=0> tau;
        real<lower=0> sigma;
        }
        model {
        lambda ~ cauchy(0, 1);
        tau ~ cauchy(0, 1);
        for (i in 1:p)
        beta[i] ~ normal(0, lambda[i] * tau);
        y ~ normal(X * beta, sigma);
        }
        """
    fit = pystan.stan(model_code=model_code, data=data, seed=5)
    beta = np.mean(fit.extract()['beta'], axis=0)
    ypred = np.dot(X_test.loc[dt], beta)
    prediction = ypred
    for i in range(len(datestotest[1:])):
        dt = datestotest[i+1]
        print 'time:',dt,'\n'
        newdata = dict(n=1, p=p, X=X.loc[datestotest[i]].values.reshape(1,-1),
                       y=[y.loc[datestotest[i]]])
        fit = pystan.stan(fit = fit, data = newdata)
        beta = np.mean(fit.extract()['beta'], axis=0)
        ypred = np.dot(X_test.loc[dt], beta)
        prediction = np.append(prediction,ypred)
    prediction = pd.Series(prediction)
    prediction.index = y_test.index
    mse = mean_squared_error(y_test,prediction)
    return((y_test,prediction),mseP)


#### These functions are for gettting the data into shape

def createPermitsData(nypermits,job_type, jobtime, column, zip1):
    nypermits_ = nypermits.loc[nypermits[column]==job_type][[jobtime, column]].groupby([jobtime], as_index=False).count()
    nypermits_.columns = [jobtime,job_type+str(zip1)]
    return nypermits_

def createPermitByNbhd(nbhd,permits):
    nypermits = permits.loc[permits.neighborhood == nbhd]
    nypermits.expiration_date = pd.to_datetime(nypermits.expiration_date)
    nypermits = nypermits.sort_values(by = 'expiration_date')
    nypermits.expiration_date = pd.to_datetime(nypermits.expiration_date).dt.strftime('%Y-%m')
    nypermits['comp'] = (nypermits['job_type'])
    nypermits['comp'] = nypermits['comp'].str.strip()
    nypermits['comp'] = [x.strip().replace(' ', '') for x in nypermits['comp']]
    
    idx = sorted(nypermits.comp.unique())
    permits_ = createPermitsData(nypermits,idx[0], 'expiration_date', 'comp', nbhd)
    for job in idx[1:]:
        permits_ = permits.merge(createPermitsData(nypermits,job,'expiration_date', 'comp', nbhd),how = 'outer',on = 'expiration_date')

    permits_ = permits_.fillna(0)
    permits_.index = permits_.expiration_date
    permits_= permits_.drop('expiration_date', axis = 1)
    permits_.index = pd.to_datetime(permits_.index)
    return permits_

def createPricesByNbhd(nyprices,nbhd):
    nyprices_ = nyprices.loc[nyprices.neighborhood == nbhd]
    nyprices_ = nyprices_.transpose()[7:].dropna(thresh=1)
    nyprices_ = nyprices_.convert_objects(convert_numeric=True).dropna(axis=0)
    nyprices_ = nyprices_.mean(axis=1)
    nyprices_.index = pd.to_datetime(nyprices_.index)
    nyprices_ = pd.DataFrame(nyprices_)
    nyprices_.columns = ['prices'+nbhd]
    return nyprices_


def createRentsByNbhd(nyprices,nbhd):
    nyprices_ = nyprices.loc[nyprices.neighborhood == nbhd]
    nyprices_ = nyprices_.transpose()[7:].dropna(thresh=1)
    nyprices_ = nyprices_.convert_objects(convert_numeric=True).dropna(axis=0)
    nyprices_ = nyprices_.mean(axis=1)
    nyprices_.index = pd.to_datetime(nyprices_.index)
    nyprices_ = pd.DataFrame(nyprices_)
    nyprices_.columns = ['rents'+nbhd]
    return nyprices_
    