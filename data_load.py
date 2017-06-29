#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:01:10 2017

@author: syedrahman
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
import os
os.chdir('/Users/syedrahman/Documents/Summer2017/Insight/Project/HousingPermitsInsight')

from helperfunctions import (createPermitsData, createPermitByNbhd)

 

dbname = 'housing'
username = 'syedrahman'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print engine

if not database_exists(engine.url):
    create_database(engine.url)

print database_exists(engine.url)


### here i download the neighborhood and zipcode and save that to an sql db

url = 'https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm'
table = pd.read_html(url, header = 0, flavor = 'bs4')
table = table[0]
table.columns = ['Borough', 'Neighborhood', 'ZIPCodes']
table.ZIPCodes.fillna(table.Neighborhood, inplace=True)
table.loc[table.Borough == 'Bronx', 'Borough'] = table.loc[table.Borough == 'Bronx', 'Neighborhood']
table.loc[table.Borough == 'Brooklyn', 'Borough'] = table.loc[table.Borough == 'Brooklyn', 'Neighborhood']
table.loc[table.Borough == 'Manhattan', 'Borough'] = table.loc[table.Borough == 'Manhattan', 'Neighborhood']
table.loc[table.Borough == 'Queens', 'Borough'] = table.loc[table.Borough == 'Queens', 'Neighborhood']
table.loc[table.Borough == 'Staten Island', 'Borough'] = table.loc[table.Borough == 'Staten Island', 'Neighborhood']
table = table.drop('Neighborhood', axis = 1)
table.columns = ['Neighborhood', 'ZIPCodes']
nbhdtable = pd.DataFrame()
for i in range(len(table)):
    for j in range(len(table.ZIPCodes.str.split()[i])):
        nbhdtable = nbhdtable.append(pd.DataFrame([(table.Neighborhood[i],table.ZIPCodes.str.split((','))[i][j].replace(' ', ''))]))
        
nbhdtable.columns = ['neighborhood', 'zip_code'] 
nbhdtable.index = range(len(nbhdtable))
nbhdtable.zip_code = nbhdtable.zip_code.convert_objects(convert_numeric=True)

nbhdtable.to_sql('nbhdtable', engine, if_exists='replace')

### now for thepermits data
permits = pd.read_csv('enigma-us.states.ny.cities.nyc.dob.permits-issued-a40d0d28f91bfd9aa4d56337dc8d6217.csv')
permits.to_sql('permits_table', engine, if_exists='replace')

### now for the home values
prices = pd.read_csv('Zip_Zhvi_AllHomes.csv') #### Prices
prices.columns = [x.lower() for x in prices.columns]
prices = prices.rename(columns = {'RegionName':'zip_code'})
prices.to_sql('home_values_table', engine, if_exists='replace')

### now for the rents
rents = pd.read_csv('Zip_Zri_AllHomesPlusMultifamily.csv') #### Rents
rents.columns = [x.lower() for x in rents.columns]
rents = rents.rename(columns = {'RegionName':'zip_code'})
rents.to_sql('rents_table', engine, if_exists='replace')

