#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:36:02 2017

@author: syedrahman
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
import os
os.chdir('/Users/syedrahman/Documents/Summer2017/Insight/Project/HousingPermitsInsight')

from helperfunctions import (createPermitsData, createPermitByNbhd, 
                             createPricesByNbhd, createRentsByNbhd)

 

dbname = 'housing'
username = 'syedrahman'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print engine

con = psycopg2.connect(database = dbname, user = username)

# query:
sql_query_nbhd = """
SELECT 
neighborhood,zip_code
FROM nbhdtable;
"""
nbhdtable = pd.read_sql_query(sql_query_nbhd,con)

# query:
sql_query_permits = """
SELECT 
a.job_type, a.permit_type,a.zip_code,a.expiration_date, b.neighborhood 
FROM permits_table a
LEFT JOIN
nbhdtable b ON a.zip_code = b.zip_code
;
"""
permits = pd.read_sql_query(sql_query_permits,con)
permits = permits.dropna(axis=0, how='any')

nbhds = np.sort(nbhdtable['neighborhood'].unique())
permitsData = createPermitByNbhd(nbhds[0],permits)
for nbhd in nbhds[1:]:
    permitsData = pd.concat([permitsData,createPermitByNbhd(nbhd,permits)], 
                                 axis = 1)

permitsData['Date'] = permitsData.index
permitsData = permitsData.dropna(axis=0, how='any')
permitsData.drop('Date', axis = 1).to_sql('final_permits_data_table',
                    engine, if_exists='replace')


# query:
sql_query_prices = """
SELECT 
*
FROM home_values_table a
LEFT JOIN
nbhdtable b ON a.zip_code = b.zip_code
;
"""
prices = pd.read_sql_query(sql_query_prices,con)
nbhds = np.sort(prices['neighborhood'].unique())
pricesData = createPricesByNbhd(prices,nbhds[0])
for nbhd in nbhds[1:]:
    pricesdata = pd.concat([pricesData,createRentsnbhd(nyrents,nbhd)], axis = 1)
    
pricesData['Date'] = pricesData.index
pricesData = pricesData.dropna(axis=0, how='any')    
pricesData.to_sql('final_prices_table',
                    engine, if_exists='replace')

# query:
sql_query_rents = """
SELECT 
*
FROM rents_table a
LEFT JOIN
nbhdtable b ON a.zip_code = b.zip_code
;
"""
rents = pd.read_sql_query(sql_query_rents,con)

nbhds = np.sort(rents['neighborhood'].unique())
rentsData = createRentsByNbhd(rents,nbhds[0])
for nbhd in nbhds[1:]:
    rentsdata = pd.concat([rentsData,createRentsnbhd(rents,nbhd)], axis = 1)
    
rentsData['Date'] = rentsData.index
rentsData = rentsData.dropna(axis=0, how='any')    
compRentsdata.to_sql('final_rents_table',
                    engine, if_exists='replace')