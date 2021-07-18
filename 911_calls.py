import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv('911.csv')

df.head()

df.info()

#top 5 zip codes

df['zip'].value_counts().head(5)

# top 5 township

df['twp'].value_counts().head(5)

#number of unique titles present in dataset

df['title'].nunique()

df['Reason'] = df['title'].apply(lambda title:title.split(':')[0])

df.head()

df['Reason'].value_counts()

sns.countplot('Reason', data = df)

type(df['timeStamp'].iloc[0])

df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['timeStamp'].head()

type(df['timeStamp'].iloc[0])

df['Hour'] = df['timeStampp'].apply(lambda time:time.hour)

df['Month'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.month)

df['DayOfWeek'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.dayofweek)

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['DayOfWeek'] = df['DayOfWeek'].map(dmap)

sns.countplot('DayOfWeek', data = df, hue = 'Reason', palette='viridis')
plt.legend(bbox_to_anchor = (1.05,1),loc = 2,  borderaxespad = 0.)

sns.countplot('Month', data = df, hue='Reason', palette='viridis')
plt.legend(bbox_to_anchor = (1.05,1),loc = 2,  borderaxespad = 0.)

by_month = df.groupby('Month').count()
by_month.head()

by_month['lat'].plot()
by_month['twp'].plot()

df['date'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.date())

df.groupby('date').count()['twp'].plot()
plt.tight_layout()

df[df['Reason']=='Traffic'].groupby('date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()

df[df['Reason']=='Fire'].groupby('date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()

df[df['Reason']=='EMS'].groupby('date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()

df1_hour_dayofweek = df.groupby(['DayOfWeek','Hour']).count()['Reason'].unstack()

sns.heatmap(df1_hour_dayofweek, cmap='viridis')
plt.tight_layout()

sns.clustermap(df1_hour_dayofweek, cmap='viridis')

df1_month_dayofweek = df.groupby(['DayOfWeek','Month']).count()['Reason'].unstack()


sns.heatmap(df1_month_dayofweek, cmap='viridis')
plt.tight_layout()

sns.clustermap(df1_month_dayofweek, cmap='viridis')