# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:00:21 2022

@author: snair
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



path = 'C:/Hertfordshire/Udemy Course/Projects/Sonar dataset'
files = glob.glob(path+'/*.csv')
dataframe = pd.DataFrame()
content=[]
for file_name in files:
    df = pd.read_csv(file_name,header=None)
    content.append(df)
    
dataframe=pd.concat(content)
print(dataframe)

dataframe.head()

dataframe.shape

## Now will check the data what are the datatypes of each feature

dataframe.info()

"""
 It's clear that it is an classification problem and all the features are numerical 
 which means there won't be any cardinality problems
 
"""

dt = dataframe.describe()

"""

As the data is scaled properly but then too if its required we can do feature scaling 
so for that I will be minmax scaling where the minimum value will be 0 and maximum will be 1

"""

data_missing = dataframe.isna()
data_missing

dt_sum = data_missing.sum()

"""

There are no missing values present in the given dataset

"""

sns.countplot(x= 60,data = dataframe)

def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('RM quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()
    

"""

This method is for detecting any outliers present in the dataset

"""

for cols in dataframe:
    diagnostic_plots(dataframe, cols)
    

"""

There are rarely any outliers present in the dataset

"""
X_train,X_test,y_train,y_test = train_test_split(dataframe.iloc[:,:-1].values,dataframe.iloc[:,-1].values,test_size = 0.2, random_state = 0)
X_train.shape,X_test.shape

"""
after splitting the data into train 
and test set the next step is to do some feature scaling in the above dataset

"""

scaler = MinMaxScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

max_scaler= scaler.data_max_

min_scaler= scaler.min_

X_train_scaled = pd.DataFrame(X_train_scaled)
X_test_scaled = pd.DataFrame(X_test_scaled)

"""

As it is an classification problem so txhere won't be any need of normalising the data so variable transformation
will be not needed and the main aim of using the min max scaler is the same.

"""

dt = dataframe.describe()

dt_scaled = X_train_scaled.describe()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train[0], ax=ax1)
sns.kdeplot(X_train[1], ax=ax1)
sns.kdeplot(X_train[2], ax=ax1)

# after scaling
ax2.set_title('After Min-Max Scaling')
sns.kdeplot(X_train_scaled[0], ax=ax2)
sns.kdeplot(X_train_scaled[1], ax=ax2)
sns.kdeplot(X_train_scaled[2], ax=ax2)
plt.show()

## Now we will perform the first machine learning algorithm to predict 
## the first one is the logistic machine learning algorithm



lr_classifier = LogisticRegression(solver='liblinear',random_state=0)
lr_classifier.fit(X_train_scaled,y_train)
y_pred = lr_classifier.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
met =[accuracy_score(y_test,y_pred), f1_score(y_test,y_pred, pos_label = 'R'), recall_score(y_test,y_pred, pos_label = 'M'), precision_score(y_test,y_pred, pos_label = 'M')]
ind = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
pd.DataFrame(data = met, index=ind, columns = ['metrics'])


## the next algorithm is the knn machine learning algorithm

print('K nearest Neighbor')

error_rate = []

for i in range(1,40):
    
    knn_classifier = KNeighborsClassifier(n_neighbors=i)
    knn_classifier.fit(X_train_scaled,y_train)
    pred_i = knn_classifier.predict(X_test_scaled)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='green', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


knn_classifier = KNeighborsClassifier(n_neighbors=4)

knn_classifier.fit(X_train_scaled,y_train)
y_pred = knn_classifier.predict(X_test_scaled)

#print('WITH K=4')
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
met = [accuracy_score(y_test,y_pred), f1_score(y_test,y_pred, pos_label = 'R'), recall_score(y_test,y_pred, pos_label = 'R'), precision_score(y_test,y_pred, pos_label = 'R')]
ind = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
pd.DataFrame(data = met, index=ind, columns = ['metrics'])

## the next machine learning algorithm is the decision tree

print('Decision Tree')

decision_tree_classifier = DecisionTreeClassifier(criterion='gini', random_state=0)
decision_tree_classifier.fit(X_train,y_train)
y_pred = decision_tree_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
met = [accuracy_score(y_test,y_pred), f1_score(y_test,y_pred, pos_label = 'R'), recall_score(y_test,y_pred, pos_label = 'R'), precision_score(y_test,y_pred, pos_label = 'R')]
ind = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
pd.DataFrame(data = met, index=ind, columns = ['metrics'])

# the next machine learning algorithm used here will be SVM

print('Support Vector Machine')

svm_classifier = SVC()
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = svm_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search.fit(X_train_scaled, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

svm_classifier = SVC(kernel = 'rbf', C = 1, gamma=0.5)
svm_classifier.fit(X_train_scaled,y_train)
y_pred = svm_classifier.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(cm)
met = [accuracy_score(y_test,y_pred), f1_score(y_test,y_pred, pos_label = 'R'), recall_score(y_test,y_pred, pos_label = 'R'), precision_score(y_test,y_pred, pos_label = 'R')]
ind = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
pd.DataFrame(data = met, index=ind, columns = ['metrics'])

"""

Summary

1. The dataset is the balanced dataset since the amount of the binary labels present in the 
dataset are almost the same.

2. There are no categorical variables so there won't any cardinality issue and any rare label problem

3. The data dosen't contain any missing values 

4. There aren't any outliers present in the dataset

5. Since the variable magnitude of data is between 0 to 2 there wasn't any need of doing any feature selection 
but for the better results the data should be standardised and it should be in the same scale atleast 
the minimum values should be 0 and maximum should be between 1.

6. The dataset doesn't contain a large amount of data so here I have proceed with the 4 machine learning 
algorithms which is logistic regression, K nearest neighbor, decision tree and support vector machine

7. I have excludeds some algorithms like naive bayes, ensemble learning as naive bayes are mostly 
used for text data and for ensemble learning we need a large amount of data to train the data.

8. The accuracy for the K Nearest Neighbor and Support Vector Machine are almost same but there are 
some difference in the precision, recall and F1 score. But if you see if the precision of the KNN algorithm is
high then the recall of SVM is high so we can't come to an conclusion like this. Henceforth if we see there is
a slight increase of f1 score of SVM as compared to the KNN. 


"""




