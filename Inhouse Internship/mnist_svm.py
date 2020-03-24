#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:38:37 2019

@author: dchhitarka
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()


n_samples = len(digits.images)
features = digits.images.reshape(n_samples,-1)
labels = digits.target

from sklearn.model_selection import train_test_split as tts
f_train, f_test, l_train, l_test = tts(features, labels, test_size=0.2, random_state=1)

from sklearn import svm

#With kernel = 'linear'
classifier = svm.SVC(kernel="linear")
classifier.fit(features[:n_samples // 2], labels[:n_samples // 2])
expected_svm = labels[n_samples // 2:]
predicted_svm = classifier.predict(features[n_samples // 2:])

from sklearn.metrics import accuracy_score, confusion_matrix
cm_svm = confusion_matrix(expected_svm, predicted_svm)
ac_svm = accuracy_score(expected_svm, predicted_svm) #0.948832
svm_train_score_on_half = classifier.score(features[:n_samples // 2], labels[:n_samples // 2]) #1.0
svm_test_score_on_half = classifier.score(features[n_samples // 2:], labels[n_samples // 2:]) #0.94438

#With split data
classifier.fit(f_train, l_train)
svm_line_pred = classifier.predict(f_test)
accuracy_score(l_test, svm_line_pred) #0.99166 for r_s = 1, 0.975 for r_s = 10
classifier.score(f_train,l_train) #1.0

#with kernel = 'rbf'
class_rbf = svm.SVC(kernel='rbf')
class_rbf.fit(features[:n_samples // 2], labels[:n_samples // 2])
expected_rbf = labels[n_samples // 2:]
predicted_rbf = class_rbf.predict(features[n_samples // 2:])

cm_svm_rbf = confusion_matrix(expected_rbf, predicted_rbf)
ac_svm_rbf = accuracy_score(expected_rbf, predicted_rbf) #0.39599
svm_rbf_train_score_on_half = class_rbf.score(features[:n_samples // 2], labels[:n_samples // 2]) #1.0
svm_rbf_test_score_on_half = class_rbf.score(features[n_samples // 2:], labels[n_samples // 2:]) #0.39599

#With split data
class_rbf.fit(f_train, l_train)
class_rbf.predict(f_test)
class_rbf.score(f_train, l_train) #1.0
accuracy_score(l_test, class_rbf.predict(f_test)) #0.43611

#With kernel = 'polynomial'
class_nl = svm.SVC(kernel='poly')
class_nl.fit(features[:n_samples // 2], labels[:n_samples // 2])
expected_nl = labels[n_samples // 2:]
predicted_nl = class_nl.predict(features[n_samples // 2:])

cm_svm_nl = confusion_matrix(expected_nl, predicted_nl)
ac_svm_nl = accuracy_score(expected_nl, predicted_nl) #0.955506
svm_nl_train_score_on_half = class_nl.score(features[:n_samples // 2], labels[:n_samples // 2]) #1.0
svm_nl_test_score_on_half = class_nl.score(features[n_samples // 2:], labels[n_samples // 2:]) #0.9555061

#With split data
class_nl.fit(f_train, l_train)
class_nl.predict(f_test)
class_nl.score(f_train, l_train) #1.0
accuracy_score(l_test, class_nl.predict(f_test)) #0.986111

from sklearn.metrics import precision_score, recall_score
print("         | Linear\t\t| Polynomial \t     | RBF ")
print("accuracy |",accuracy_score(l_test, svm_line_pred),"  |",accuracy_score(l_test, class_nl.predict(f_test)),"|",accuracy_score(l_test, class_rbf.predict(f_test)))
print("precision|",precision_score(l_test, svm_line_pred, average="weighted"),"  |",precision_score(l_test, class_nl.predict(f_test), average="weighted"),"|",precision_score(l_test, class_rbf.predict(f_test), average="weighted"))
print("recall   |",recall_score(l_test, svm_line_pred, average="weighted"),"  |",recall_score(l_test, class_nl.predict(f_test), average="weighted"),"|",recall_score(l_test, class_rbf.predict(f_test), average="weighted"))
"""
         | Linear               | Polynomial         | RBF 
accuracy | 0.9916666666666667   | 0.9861111111111112 | 0.4361111111111111
precision| 0.9920138888888889   | 0.9866452991452991 | 0.9273962804005723
recall   | 0.9916666666666667   | 0.9861111111111112 | 0.4361111111111111
"""



#print("precision|",precision_score(l_test, svm_line_pred, average="macro"),"  |",precision_score(l_test, class_nl.predict(f_test), average="macro"),"|",precision_score(l_test, class_rbf.predict(f_test), average="macro"))
#precision| 0.9908088235294118   | 0.9857692307692307 | 0.9128755364806868

#print("precision|",precision_score(l_test, svm_line_pred, average="micro"),"  |",precision_score(l_test, class_nl.predict(f_test), average="micro"),"|",precision_score(l_test, class_rbf.predict(f_test), average="micro"))
#precision| 0.9916666666666667   | 0.9861111111111112 | 0.4361111111111111


