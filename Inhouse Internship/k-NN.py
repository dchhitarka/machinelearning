# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:19:31 2019

@author: sharm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Social_Network_Ads .csv")
features=df.iloc[:,2:-1].values
labels=df.iloc[:,-1].values

#splitting dataset

from sklearn.cross_validation import train_test_split
f_train, f_test, l_train, l_test= train_test_split(features, labels, test_size=0.25, random_state=0) 

# feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
f_train=sc.fit_transform(f_train)
f_test=sc.fit_transform(f_test)


# fitting k-nn to training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(f_train,l_train)

pred=classifier.predict(f_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(l_test,pred)
     
score=classifier.score(f_test,l_test)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
features_set, labels_set = f_train, l_train
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(features1, features2, classifier.predict(np.array([features1.ravel(), features2.ravel()]).T).reshape(features1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
# Visualising the Training set results
from matplotlib.colors import ListedColormap
features_set, labels_set = f_test, l_test
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(features1, features2, classifier.predict(np.array([features1.ravel(), features2.ravel()]).T).reshape(features1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

