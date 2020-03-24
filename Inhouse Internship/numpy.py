# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:15:17 2019

@author: sharm
"""

import numpy as np
incomes=np.random.normal(27000,15000,10000)
np.mean(incomes)

import matplotlib.pyplot as plt
plt.hist(incomes,50)
plt.show()

np.median(incomes)

incomes=np.append(incomes,[1000000000])

np.median(incomes)

np.mean(incomes)

incomes.std()
incomes.var()