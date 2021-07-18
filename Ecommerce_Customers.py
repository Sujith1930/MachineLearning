import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('Ecommerce Customers')

dataset.head()

dataset.info()


des = dataset.describe()

sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = dataset)

sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent', data = dataset)

sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent', kind = 'hex' , data = dataset)

sns.pairplot(dataset)

sns.regplot(x = 'Length of Membership', y = 'Yearly Amount Spent', data = dataset) # instead of regplot used inplot


X = dataset.iloc[:,3:-1]

y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=101)

lmr = LinearRegression()

lmr.fit(X_train, y_train)

lmr.coef_

lmr.intercept_

predictions = lmr.predict(X_test)

sns.scatterplot(y_test, predictions)
plt.xlabel('y_test')
plt.ylabel('predictions')


print('MAE', metrics.mean_absolute_error(y_test, predictions))
print('MSE', metrics.mean_squared_error(y_test, predictions))
print ('RMSE', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Average', round(np.mean(dataset['Yearly Amount Spent']),2))
print ('R Square', metrics.r2_score(y_test, predictions))

sns.displot((y_test - predictions), bins = 50, kde = True)

coefficients = pd.DataFrame(lmr.coef_, X.columns)
coefficients.columns = ['Columns']
coefficients

#Holding all other features fixed, a 1 unit increase in **Avg. Session Length** is associated with an **increase of 25.98 total dollars spent**.
#Holding all other features fixed, a 1 unit increase in **Time on App** is associated with an **increase of 38.59 total dollars spent**.
#Holding all other features fixed, a 1 unit increase in **Time on Website** is associated with an **increase of 0.19 total dollars spent**.
#Holding all other features fixed, a 1 unit increase in **Length of Membership** is associated with an **increase of 61.27 total dollars spent**.