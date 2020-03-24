#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:58:46 2019

@author: dchhitarka
"""

import mnist_knn
import mnist_svm
import mnist_logistic_reg



from sklearn.metrics import precision_score, recall_score, accuracy_score
print("         | KNN\t\t\t| SVM (linear)       | Logistic regression ")
print("accuracy |",accuracy_score(mnist_knn.l_test, knn.predict(mnist_knn.f_test)),"  |",accuracy_score(l_test, svm_line_pred),"|",accuracy_score(l_test, lr_pred))
print("precision|",precision_score(l_test, knn.predict(mnist_knn.f_test), average="weighted"),"  |",precision_score(l_test, svm_line_pred, average="weighted"),"|",precision_score(l_test, lr_pred, average="weighted"))
print("recall   |",recall_score(l_test, knn.predict(mnist_knn.f_test), average="weighted"),"  |",recall_score(l_test, svm_line_pred, average="weighted"),"|",recall_score(l_test, lr_pred, average="weighted"))

"""
         | KNN                  | SVM (linear)        | Logistic regression 
accuracy | 0.9555555555555556   | 0.9916666666666667  | 0.9694444444444444
precision| 0.08477379115030656   | 0.9920138888888889 | 0.9702107540010767
recall   | 0.08333333333333333   | 0.9916666666666667 | 0.9694444444444444
"""