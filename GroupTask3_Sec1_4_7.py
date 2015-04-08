## group task practice
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pytz
from datetime import datetime, timedelta
from dateutil import parser
from scipy.stats import ttest_ind
import statsmodels.api as sm


main_dir = "c:/Users/Dennis Bartlett/Desktop/PubPol590/"

root = main_dir + "Data/Logit/"
df = pd.read_csv(root + "allocation_subsamp.csv")

vec_ctrl = df[df['tariff'] == 'E']
vec_ctrl = vec_ctrl['ID']

vec_A1 = df[df['tariff'] == 'A']
vec_A1 = vec_A1[df['stimulus'] == '1']
vec_A1 = vec_A1['ID']

vec_A3 = df[df['tariff'] == 'A']
vec_A3 = vec_A3[df['stimulus'] == '3']
vec_A3 = vec_A3['ID']

vec_B1 = df[df['tariff'] == 'B']
vec_B1 = vec_B1[df['stimulus'] == '1']
vec_B1 = vec_B1['ID']

vec_B3 = df[df['tariff'] == 'B']
vec_B3 = vec_B3[df['stimulus'] == '3']
vec_B3 = vec_B3['ID']

np.random.seed(1789)

# SELECT RANDOM SAMPLE IDS FROM EACH CATEGORY
samp_ctrl = np.random.choice(vec_ctrl,size=300,replace=False)
samp_A1 = np.random.choice(vec_A1,size=150,replace=False)
samp_A3 = np.random.choice(vec_A3,size=150,replace=False)
samp_B1 = np.random.choice(vec_B1,size=50,replace=False)
samp_B3 = np.random.choice(vec_B3,size=50,replace=False)

# TRANSFORM TO DATAFRAMES AND CONCAT
df_ctrl = DataFrame(samp_ctrl)
df_A1 = DataFrame(samp_A1)
df_A3 = DataFrame(samp_A3)
df_B1 = DataFrame(samp_B1)
df_B3 = DataFrame(samp_B3)
df2 = pd.concat([df_ctrl,df_A1, df_A3,df_B1,df_B3])

# CLEAN UP DATAFRAME
df2.reset_index(inplace=True)
df2.columns = ['sample','ID']
del df2['sample']

# CLEAR UNWANTED VARIABLES
del df_A1, df_A3, df_B1, df_B3, df_ctrl, samp_A1, samp_A3, samp_B1, samp_B3, samp_ctrl
del vec_A1, vec_A3, vec_B1, vec_B3, vec_ctrl

df_consump = pd.read_csv(root + "kwh_redux_pretrial.csv", parse_dates=[2], date_parser=np.datetime64)

df_merge = pd.merge(df2, df_consump)

# CHECK RESULTS - used df_merge.drop_duplicates(cols='ID') to confirm that 
# there are still 700 values for ID
# df_consump starts at 14 million rows, df_merge is 5.7 million rows

df_merge['year'] = df_merge['date'].apply(lambda x: x.year)
df_merge['month'] = df_merge['date'].apply(lambda x: x.month)
df_merge['day'] = df_merge['date'].apply(lambda x: x.day)
del df_merge['date']

df_merge['month'] = ['0' + str(v) if v <10 else str(v) for v in df_merge['month']]

# GROUP BY ID NUMBER AND AGGREGATE kWh NUMBERS
grp = df_merge.groupby(['ID', 'month'])
df_agg = grp['kwh'].sum().reset_index()

# PIVOT DATA
df_agg = df_agg.pivot(index='ID', columns='month', values = 'kwh')
df_agg.reset_index(inplace=True) #deletes the null row at the top
df_agg.columns.name = None #delete the first column header
df_agg.columns = ['ID', 'kWh_Jul', 'kWh_Aug','kWh_Sep','kWh_Oct','kWh_Nov','kWh_Dec']

# MERGE ALLOCATION DATA TO WIDE DATASET
df_final = pd.merge(df_agg, df)
df_final['tarstim'] = df_final['tariff'] + df_final['stimulus']
del df_final['code']
del df_final['tariff']
del df_final['stimulus']

dummy_ranks = pd.get_dummies(df_final['tarstim'], prefix='tarstim')
cols_to_keep = ['ID', 'kWh_Jul', 'kWh_Aug','kWh_Sep','kWh_Oct','kWh_Nov','kWh_Dec']
df_logit = df_final[cols_to_keep].join(dummy_ranks)

## SET UP THE DATA FOR LOGIT ------
kwh_cols = [v for v in df_logit.columns.values if v.startswith('kWh')]
df_logit_A1 = df_logit[(df_logit['tarstim_A1'] == 1) | (df_logit['tarstim_EE'] == 1)]
df_logit_A3 = df_logit[(df_logit['tarstim_A3'] == 1) | (df_logit['tarstim_EE'] == 1)]
df_logit_B1 = df_logit[(df_logit['tarstim_B1'] == 1) | (df_logit['tarstim_EE'] == 1)]
df_logit_B3 = df_logit[(df_logit['tarstim_B3'] == 1) | (df_logit['tarstim_EE'] == 1)]

## SET UP Y, X FOR A1 VS CONTROL
y_A1 = df_logit_A1['tarstim_A1']
x_A1 = df_logit_A1[kwh_cols]
x_A1 = sm.add_constant(x_A1)

## SET UP Y, X FOR A3 VS CONTROL
y_A3 = df_logit_A3['tarstim_A3']
x_A3 = df_logit_A3[kwh_cols]
x_A3 = sm.add_constant(x_A3)

## SET UP Y, X FOR B1 VS CONTROL
y_B1 = df_logit_B1['tarstim_B1']
x_B1 = df_logit_B1[kwh_cols]
x_B1 = sm.add_constant(x_B1)

## SET UP Y, X FOR B3 VS CONTROL
y_B3 = df_logit_B3['tarstim_B3']
x_B3 = df_logit_B3[kwh_cols]
x_B3 = sm.add_constant(x_B3)

## LOGIT A1 VS CONTROL-------------
logit_model = sm.Logit(y_A1, x_A1)
logit_results = logit_model.fit()
print(logit_results.summary())

## LOGIT A3 VS CONTROL-------------
logit_model = sm.Logit(y_A3, x_A3)
logit_results = logit_model.fit()
print(logit_results.summary())

## LOGIT B1 VS CONTROL-------------
logit_model = sm.Logit(y_B1, x_B1)
logit_results = logit_model.fit()
print(logit_results.summary())

## LOGIT B3 VS CONTROL-------------
logit_model = sm.Logit(y_B3, x_B3)
logit_results = logit_model.fit()
print(logit_results.summary())
