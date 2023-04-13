
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yfinance'])

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from tabulate import tabulate
import pandas as pd
import yfinance as yf
plt.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.linewidth'] = 2 
plt.rcParams.update({"text.usetex": True})



vix = pd.read_csv('VIX_History.csv', index_col=[0])
spx = yf.download('^FCHI')

import os
os.getcwd()

vix.index = pd.to_datetime(vix.index)
df = pd.concat([vix['CLOSE'], spx['Close']], axis = 1).dropna()
df.columns = ['vix', 'spx']

df.head()
df.tail()

df.to_csv('data.csv')
plt.plot(df['spx'], df['vix'], '.', color = 'firebrick')
plt.show()

df_plot = df[(df.index.year >= 2000) & (df.index.year <= 2018)]

fig,ax = plt.subplots()
l1, =ax.plot(df_plot.index, df_plot['spx'], color = 'firebrick', label = 'spx')
ax.set_ylabel('SPX', color = 'firebrick')
[t.set_color('firebrick') for t in ax.yaxis.get_ticklabels()]
ax2 = ax.twinx()
l2, = ax2.plot(df_plot.index, df_plot['vix'], color = 'navy', label = 'spx')
# make a plot with different y-axis using second axis object
ax2.set_ylabel("VIX", color = "navy")
[t.set_color('navy') for t in ax2.yaxis.get_ticklabels()]
plt.legend([l1, l2], ['SPX', 'VIX'])
plt.show()
fig.savefig('ouss.pdf', format = 'pdf')