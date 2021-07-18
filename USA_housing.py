import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('USA_Housing.csv')

dataset.head()

dataset.info()

des = dataset.describe()

dataset.columns

sns.pairplot(dataset)

sns.distplot(dataset['Price'])

sns.heatmap(dataset.corr())


X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = dataset['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.intercept_)

coeff = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

#Residual histogram

sns.displot((y_test - predictions), bins=50)


print ('MAE ', round (metrics.mean_absolute_error(y_test, predictions),3))
print ('MSE ', round (metrics.mean_squared_error(y_test, predictions),2))
print ('RMSE ', round (np.sqrt(metrics.mean_squared_error(y_test, predictions)),2))

print (round(dataset['Price'].mean(),2))

print('R2 :-', metrics.r2_score(y_test, predictions))