"""9/9 pts"""

from __future__ import division
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from pandas import Series, DataFrame

main_dir = '/Users/dnoriega/Dropbox/pubpol590_sp15/data_sets/CER/tasks/4_task_data/'

# CHANGE WORKING DIRECTORY (wd)
os.chdir(main_dir)
from logit_functions import *

# IMPORT DATA ------
df = pd.read_csv(main_dir + 'task_4_kwh_w_dummies_wide.csv')
df = df.dropna(axis=0, how='any') #defaults

# GET TARIFFS -------
tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()

# RUN LOGIT ---------
drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(drop, axis=1)

for i in tariffs:
    for j in stimuli:
        # dummy vars must start with "D_" and consumption vars with "kwh_"
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc=False)

# QUICK MEANS COMPARISON WITH T-TEST BY HAND ---------
# create means
df_mean = df_logit.groupby('tariff').mean().transpose() # just tariff is sufficient because only have one treatment group
df_mean.C - df_mean.E

# do a t-test "by hand"
df_s = df_logit.groupby('tariff').std().transpose() # standard errors
df_n = df_logit.groupby('tariff').count().transpose().mean() # sample sizes
top = df_mean['C'] - df_mean['E']
bottom = np.sqrt(df_s['C']**2/df_n['C'] + df_s['E']**2/df_n['E'])
tstats = top/bottom
sig = tstats[np.abs(tstats) > 2]
sig.name = 't-stats'

##### SECTION 2 -------------------

# create a column with p hat values using the predict() method
df_logit['p_val']=logit_results.predict()

# create a column to identify 'treated' IDs, where 1= treated and 0 = control
df_logit['trt'] = 0 + (df_logit['tariff'] == 'C')

# generate a column of propensity score weights
# use equation w = sqrt((D/p) + (1-D)/(1-p))
df_logit['w'] = np.sqrt(df_logit['trt']/df_logit['p_val']
+ (1-df_logit['trt'])/(1-df_logit['p_val']))

# simplify desired results into separate df
df_w = df_logit[['ID', 'trt','w']]

## SECTION 3 ----------

# change working directory (wd)
os.chdir(main_dir)
from fe_functions import *

# import data ------
df_fe = pd.read_csv(main_dir + 'task_4_kwh_long.csv')
df_fe = df_fe.dropna(axis=0, how='any') #defaults

# merge with weight data -----
df_fe = pd.merge(df_w, df_fe)

# create a treatment/trial interaction variable 0 = pre-trial, 1= trial
df_fe['trial'] = 0 + (df_fe['year'] == 2010)
df_fe['treattrial'] = df_fe['trt'] * df_fe['trial']

# create 'log plus one' column for kwh vector
df_fe['log_kwh']=(df_fe['kwh'] + 1).apply(np.log)

# create a year-month column
df_fe['mo_str'] = ['0' + str(v) if v <10 else str(v) for v in df_fe['month']]
df_fe['ym'] = df_fe['year'].apply(str) + "_" + df_fe['mo_str']

# set up regression variables
y = df_fe['log_kwh']
p = df_fe['trial']
tp = df_fe['treattrial']
w = df_fe['w']
mu = pd.get_dummies(df_fe['ym'], prefix = 'ym').iloc[:,1:-1]
x = pd.concat([tp, p, mu], axis=1)

# demean y and x using demean
ids = df_fe['ID']
y = demean(y, ids)
x = demean(x, ids)

## fixed effects WITHOUT WEIGHTS
fe_model = sm.OLS(y, x) # linearly prob model
fe_results = fe_model.fit() # get the fitted values
print(fe_results.summary()) # print pretty results (no results given lack of obs)

# fixed effects WITH WEIGHTS
## apply weights to data
y = y*w # weight each y
nms = x.columns.values # savxe column names
x = np.array([x*w for k, x in x.iteritems()]) # weight each X value
x = x.T # transpose (necessary as arrays create "row" vectors, not column)
x = DataFrame(x, columns = nms) # update to dataframe; use original names

fe_w_model = sm.OLS(y, x) # linearly prob model
fe_w_results = fe_w_model.fit() # get the fitted values
print(fe_w_results.summary()) # print pretty results (no results given lack of obs)