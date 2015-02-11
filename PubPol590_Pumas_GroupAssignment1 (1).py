from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

## PATHING -------------------------
main_dir = "/Users/mirandamarks/Documents/Documents/My Documents/Duke/Spring 2015/PubPol590/"
root = main_dir + "CER_both/Data/"
assign_file = "SME_Residential_Allocations.csv"

## IMPORT DATA ---------------------
paths = [os.path.join(root, v) for v in os.listdir(root) if v.startswith("File")]
missing = ['-', ' ', 'NA', '.', 'null', '9999999']
list_of_dfs = [pd.read_table(v, names = ['ID', 'Date', 'kWh'], skiprows = 6000000, nrows = 1500000, na_values = missing, sep = " ") for v in paths]
df_assign = pd.read_csv(root + assign_file, usecols = [0,1,2,3,4], na_values = missing)

## STACK AND MERGE -------------------
df = pd.concat(list_of_dfs, ignore_index = True)
df = pd.merge(df, df_assign)

## drop old dataframes
del list_of_dfs, df_assign

# CLEAN DATA ---------------------

## drop duplicates
df.drop_duplicates(['ID', 'Date'])

## create separate day and hour columns
df['hour'] = df['Date'] % 100 # extract last two digits of date as hour
df['day'] = (df['Date'] - df['hour']) / 100 # extract first three digits of date as day