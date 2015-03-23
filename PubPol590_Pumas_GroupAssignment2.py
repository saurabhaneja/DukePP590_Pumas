""" NOT THE MOST EFFICIENT CODE (DUE TO ONE LINE) BUT REALLY PRETTY!"""
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

## PATHING -------------------------
main_dir = "/Users/dnoriega/Dropbox/pubpol590_sp15/data_sets/CER/"
root = main_dir + "cooked/"
assign_file = "SME and Residential allocations.csv"
time_file = "timeseries_correction.csv"

## IMPORT DATA ---------------------
paths = [os.path.join(root, v) for v in os.listdir(root) if v.startswith("File")]
missing = ['-', ' ', 'NA', '.', 'null', '9999999']
df = pd.concat([pd.read_table(v, names = ['ID', 'date_cer', 'kWh'], na_values = missing, sep = " ") for v in paths], ignore_index = True)

# CLEAN DATA ---------------------

## create separate day and hour columns
df['hour_cer'] = df['date_cer'] % 100 # extract last two digits of date as hour
df['day_cer'] = (df['date_cer'] - df['hour_cer']) / 100 # extract first three digits of date as day

## import timeseries correction to fix date issues
df_time = pd.read_csv(root + time_file, header=0, usecols=[1,2,3,4,5,6,7,8,9,10])


# IMPORT ASSIGNMENT DATA -----------
df_assign = pd.read_csv(root + assign_file, usecols = [0,1,2,3,4], header = True, names = ['ID', 'Code', 'res_tariff', 'res_stimulus', 'SME'], na_values = missing) # keep only first 5 columns

## drop all but residential homes in control or bi-monthly only stimulus and tariff A groups
grp = df_assign.groupby(['Code', 'res_tariff', 'res_stimulus'])
# a is a list of lists of indices that satisfy the conditions; a[0] are the 1,E,E rows and a[1] are the 1,A,1 rows
a = [v for k, v in grp.groups.iteritems() if (k[0] == 1 and ((k[1] == 'E') or (k[1] == 'A' and k[2] == '1')))]
# the .ix allows us to extract multiple rows from a list of indices, a[0] and a[1]
df_assign = pd.concat([df_assign.ix[a[0]], df_assign.ix[a[1]]])

# MERGE DATA WITH ASSIGNMENT ----------------
df = pd.merge(df, df_assign) # default merges on common 'ID' key

"""MOVED FROM ABOVE -- THIS WILL GREATLY CUT DOWN ON CODE RUNNING TIME"""
df = pd.merge(df, df_time) # default merges on common hour_cer and day_cer keys

## drop old dataframes
del df_assign

# AGGREGATE BY DAY -------------

grp = df.groupby(['year', 'month', 'day', 'ID', 'res_stimulus'])
agg = grp['kWh'].sum()

## reset index
agg = agg.reset_index()
grp1 = agg.groupby(['year', 'month', 'day', 'res_stimulus'])

## split up T/C
trt = {(k[0], k[1], k[2]): agg.kWh[v].values for k, v in grp1.groups.iteritems() if k[3] == '1'}
ctrl = {(k[0], k[1], k[2]): agg.kWh[v].values for k, v in grp1.groups.iteritems() if k[3] == 'E'}
keys = ctrl.keys()

# tstats and pvals
tstats = DataFrame([(k, np.abs(float(ttest_ind(trt[k], ctrl[k], equal_var=False)[0]))) for k in keys], columns=['ymd', 'tstat'])
pvals = DataFrame([(k, np.abs(float(ttest_ind(trt[k], ctrl[k], equal_var=False)[1]))) for k in keys], columns=['ymd', 'pval'])
t_p = pd.merge(tstats, pvals)

# sort and reset
t_p.sort(['ymd'], inplace=True)
t_p.reset_index(inplace=True, drop=True)

# PLOTTING BY DAY ----------------
fig1 = plt.figure() # initialize plot
ax1 = fig1.add_subplot(2,1,1) # 2 rows, 1 col, first plot
ax1.plot(t_p['tstat']) # only one column of data plots data as y and uses index as x
ax1.axhline(2, color = 'r', linestyle = '--') # horizontal line
ax1.axvline(171, color = 'g', linestyle = '--') # vertical line - trt began 1Jan2010
ax1.set_title('t-stats over-time (daily)')
ax2 = fig1.add_subplot(2,1,2)
ax2.plot(t_p['pval'])
ax2.axhline(0.05, color = 'r', linestyle = '--')
ax2.axvline(171, color = 'g', linestyle = '--') # vertical line - trt began 1Jan2010
ax2.set_title('p-values over-time(daily)')

# AGGREGATE BY MONTH -------------

grp2 = df.groupby(['year', 'month', 'ID', 'res_stimulus'])
agg2 = grp2['kWh'].sum()

## reset index
agg2 = agg2.reset_index()
grp3 = agg2.groupby(['year', 'month', 'res_stimulus'])

## split up T/C
trt2 = {(k[0], k[1]): agg2.kWh[v].values for k, v in grp3.groups.iteritems() if k[2] == '1'}
ctrl2 = {(k[0], k[1]): agg2.kWh[v].values for k, v in grp3.groups.iteritems() if k[2] == 'E'}
keys2 = ctrl2.keys()

# tstats and pvals
tstats2 = DataFrame([(k, np.abs(float(ttest_ind(trt2[k], ctrl2[k], equal_var=False)[0]))) for k in keys2], columns=['ym', 'tstat'])
pvals2 = DataFrame([(k, np.abs(float(ttest_ind(trt2[k], ctrl2[k], equal_var=False)[1]))) for k in keys2], columns=['ym', 'pval'])
t_p2 = pd.merge(tstats2, pvals2)

# sort and reset
t_p2.sort(['ym'], inplace=True)
t_p2.reset_index(inplace=True, drop=True)

# PLOTTING BY MONTH ----------------
fig2 = plt.figure() # initialize plot
ax3 = fig2.add_subplot(2,1,1) # 2 rows, 1 col, first plot
ax3.plot(t_p2['tstat']) # only one column of data plots data as y and uses index as x
ax3.axhline(2, color = 'r', linestyle = '--') # horizontal line
ax3.axvline(6, color = 'g', linestyle = '--') # vertical line - trt began 1Jan2010
ax3.set_title('t-stats over-time (monthly)')
ax4 = fig2.add_subplot(2,1,2)
ax4.plot(t_p2['pval'])
ax4.axhline(0.05, color = 'r', linestyle = '--')
ax4.axvline(6, color = 'g', linestyle = '--') # vertical line - trt began 1Jan2010
ax4.set_title('p-values over-time(monthly)')

