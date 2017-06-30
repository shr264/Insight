#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 01:11:14 2017

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

sql_query_prices = """
    SELECT
    *
    FROM final_prices_table;
    """
compPricesdata = pd.read_sql_query(sql_query_prices,con)

sql_query_permits = """
    SELECT
    *
    FROM final_permits_data_table;
    """
compPermitsdata = pd.read_sql_query(sql_query_permits,con)

### this is the differencing
yyP = compPricesdata.drop(['Unnamed: 0'], axis = 1)
yyPdates = yyP.Date
yyP = yyP.drop(['Date'], axis = 1)
yyPcolumns = yyP.columns

### Differencing Y
yyP.index = yyPdates
yyP.columns = yyPcolumns
y1P = yyP.shift(1).fillna(0)
yP = yyP-y1P

### now the autoregressive terms
y11 = yP.shift(1).fillna(0)
y21 = y11.shift(1).fillna(0)
y11.columns = [(x+'_1') for  x in y11.columns]
y21.columns = [(x+'_1') for  x in y21.columns]

### similar for permits data
X = compPermitsdata.drop(['expiration_date'],axis = 1).fillna(0)
Xdates = X.Date
X = X.drop(['Date'],axis = 1).fillna(0)
Xcolumns = X.columns

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

### making sure we have only common dates
i1 = yP.index
i2 = X.index
yP = yP[i1.isin(i2)]
y1P = y1P[i1.isin(i2)]
X = X[i2.isin(i1)]
    
alpha = np.linspace(0.8,2.5,20)
mseP = findLassoAlpha(alpha[0],yP,X)
for i in range(len(alpha[1:])):
    mseP = np.append(mseP,findLassoAlpha(alpha[i],yP,X))  

alphastarP = alpha[np.where(mseP == np.amin(mseP))[0][0]]  


pricePredsVAR = findLassoPreds(alphastarP,yP,X) 

np.sqrt(mean_squared_error(pricePredsVAR[0][0]['pricesGramercy Park and Murray Hill'],
 pricePredsVAR[0][1]['pricesGramercy Park and Murray Hill']))

prophetPreds = findProphetPreds(yyP,X, categ = 'prices')
np.sqrt(prophetPreds[1])
    
rnnPreds = findRNNPreds(yP,X, categ = 'prices')   
np.sqrt(rnnPreds[1])

bayesPreds = findBayesHSPred(yP,X, categ = 'prices', 
                             nbhd = 'Gramercy Park and Murray Hill')

methods = ['l1-VAR','Prophet','RNN','Bayes']
mseVals = [np.sqrt(mean_squared_error(pricePredsVAR[0][0]['pricesGramercy Park and Murray Hill'],
 pricePredsVAR[0][1]['pricesGramercy Park and Murray Hill'])),
    np.sqrt(prophetPreds[1]), 
    np.sqrt(rnnPreds[1]),
    np.sqrt(bayesPreds[1])]
mseVals = pd.DataFrame([mseVals])
mseVals.columns = methods

print '------------------------------'
print mseVals