#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:00:48 2017

@author: syedrahman
"""

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import requests
from io import BytesIO
import os
from sklearn.linear_model import Lasso, LassoCV,  MultiTaskLassoCV, MultiTaskLasso
from pandas.tools.plotting import autocorrelation_plot
import networkx as nx
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import pystan
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

from helperfunctions import (findLassoPreds, findLassoAlpha, findProphetPreds, 
                             findRNNPreds, findBayesHSPred)

dbname = 'housing'
username = 'syedrahman'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print engine

con = psycopg2.connect(database = dbname, user = username)

sql_query_nbhd = """
    SELECT
    neighborhood,zip_code
    FROM nbhdtable;
    """
nbhdtable = pd.read_sql_query(sql_query_nbhd,con)

sql_query_rents = """
    SELECT
    *
    FROM final_rents_table;
    """
compPermitsdata = pd.read_sql_query(sql_query_rents,con)

sql_query_permits = """
    SELECT
    *
    FROM final_permits_data_table;
    """
compPermitsdata = pd.read_sql_query(sql_query_permits,con)

### this is the differencing
yy = compRentsdata.drop(['Unnamed: 0'], axis = 1)
yy.index = yy.Date
yy = yy.drop(['Date'], axis = 1)
yy.to_csv('yyR.csv')
y1 = yy.shift(1).fillna(0)
y = yy-y1

### now the autoregressive terms
y11 = y.shift(1).fillna(0)
y21 = y11.shift(1).fillna(0)
y11.columns = [(x+'_1') for  x in y11.columns]
y21.columns = [(x+'_1') for  x in y21.columns]

### similar for permits data
X = compPermitsdata.drop(['expiration_date'],axis = 1).fillna(0)
Xdates = X.Date
X = X.drop(['Date'],axis = 1).fillna(0)
Xcolumns = X.columns

### here we are setting up the data from VAR - more details on the website
Y_var = pd.concat([y,X], axis = 1)
Y_var_1 = Y_var.shift(1).fillna(0)
Y_var_2 = Y_var_1.shift(1).fillna(0)
Z_ones = pd.DataFrame(np.ones(Y_var_1.shape[0]))
Z_ones.index = Y_var_1.index
Z_var = pd.concat([Z_ones,Y_var_1,Y_var_1], axis = 1)

### scaling the X data
Xscaler = StandardScaler()
X = Xscaler.fit_transform(X)
X = pd.DataFrame(X)
X.index = Xdates
X.columns = Xcolumns
X_1 = X.shift(1).fillna(0)
X_diff = X-X_1
X_diff_1 = X_diff.shift(1).fillna(0)
X_diff_2 = X_diff_1.shift(1).fillna(0)
X = pd.concat([X_diff,X_diff_1,X_diff_2,y11,y21], axis = 1).dropna(axis = 0)
X.head()

i1 = y.index
i2 = X.index
y = y[i1.isin(i2)]
y1 = y1[i1.isin(i2)]
yyR = yy[i1.isin(i2)]
    
alpha = np.linspace(3,20, 20)
mse = findLassoAlpha(alpha[0],y,X)
for i in range(len(alpha[1:])):
    mse = np.append(mse,findLassoAlpha(alpha[i],Y_var,Z_var))  

alphastar = alpha[np.where(mse == np.amin(mse))[0][0]]  

rentPredsVAR = findLassoPreds(alphastar,Y_var,Z_var)

prophetRPreds = findProphetPreds(yyR,X, categ = 'rents')
    
rnnRPreds = findRNNPreds(y,X, categ = 'rents')    

bayesRPreds = findBayesHSPred(y,X, categ = 'rents')

methods = ['l1-VAR','Prophet','RNN','Bayes']
mseVals = [np.sqrt(mean_squared_error(rentPredsVAR[0][0]['rentsGramercy Park and Murray Hill'],
 rentPredsVAR[0][1]['rentsGramercy Park and Murray Hill'])),
    np.sqrt(prophetRPreds[1]), 
    np.sqrt(rnnRPreds[1]),
    np.sqrt(bayesRPreds[1])]
mseVals = pd.DataFrame([mseVals])
mseVals.columns = methods

print '------------------------------'
print mseVals



