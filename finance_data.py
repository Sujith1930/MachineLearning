import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pandas_datareader import data, wb
from datetime import datetime
import plotly
import cufflinks as cf
cf.go_offline()

start = datetime(2006, 1, 1)
end = datetime(2016,1,1)

df = pd.read_pickle('all_banks')

df.head()

df.info()

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

#for tick in tickers:
#    print (tick, df[tick]['Close'].max())

df.xs(key = 'Close', axis = 1, level = 'Stock Info').max()

df.xs(key = 'Close', axis = 1, level = 'Stock Info').min()

returns = pd.DataFrame()

for tick in tickers:
    returns[tick+ '  Return'] = df[tick]['Close'].pct_change()
returns.head()

sns.pairplot(returns[1:])

returns.idxmin() # works like argmin and returns data for the entire column

returns.idxmax()

returns.std()
# it's clear that the stock of city bank is very risky

returns.head()

returns['2015-01-1':'2015-12-31'].std()
# the stocks of WFC is less risky as compared to the other banks stocks


sns.displot(returns['2015-01-1':'2015-12-31']['MS  Return'], color = 'green', bins = 50, kde = True)

sns.distplot(returns['2008-01-1':'2008-12-31']['C  Return'], color = 'red', bins = 50, kde = True)

for tick in tickers:
    df[tick]['Close'].plot(label = tick, figsize = (12,4))
plt.legend()

df.xs(key = 'Close', axis = 1, level = 'Stock Info').plot(label = tick, figsize = (12,4))
plt.legend()

df.xs(key = 'Close', axis = 1, level = 'Stock Info').iplot()


df['BAC']['Close']['2008-01-1':'2009-01-1'].rolling(window= 30).mean().plot(label ='30 day Mov')
df['BAC']['Close']['2008-01-1':'2009-01-1'].plot(label ='BAC Close')
plt.legend()

sns.heatmap(df.xs(key='Close',axis = 1, level = 'Stock Info').corr(), annot = True)