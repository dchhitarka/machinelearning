# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:19:22 2018

@author: sharm
"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


dataset=pd.read_csv("Match_Making.csv")
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,-1].values

#splitting data set into the Training set and Test set
from sklearn.cross_validation import train_test_split
f_train, f_test, l_train, l_test= train_test_split(features, labels, test_size=0.3, random_state=0)

#feature scaling
""""from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
f_train= sc.fit_transform(f_train)
f_test= sc.fit_transform(f_test)"""


# fitting kernel SVM to the training set
#kernels: linear, poly, rbf
from sklearn.svm import SVC
classifier=SVC(kernel="rbf", random_state=0)
classifier.fit(f_train, l_train)


#predicting the test result
labels_pred=classifier.predict(f_test)


# making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(l_test, labels_pred)



# Visualising the Training set results
from matplotlib.colors import ListedColormap
features_set, labels_set = f_train, l_train
X1, X2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
features_set, labels_set = f_test, l_test
X1, X2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



