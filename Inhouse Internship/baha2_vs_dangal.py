#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:41:47 2019

@author: dchhitarka

Import Bahubali2vsDangal.csv file.

It contains Data of Day wise collections of the movies Bahubali 2 and Dangal (in crores) for the first 9 days. 
Now, you have to write a python code to predict which movie would collect more on the 10th day.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('Bahubali2_vs_Dangal.csv')
df.isnull().any(axis=0)
df.columns
#x = df['Days']
#yb = df['Bahubali_2_Collections_Per_day']
#yd = df['Dangal_collections_Per_day']
days = df[['Days']]
bahu = df.iloc[:,1]
dang = df.iloc[:,2]


from sklearn.linear_model import LinearRegression

lb = LinearRegression()
lb.fit(days, bahu)

ldang = LinearRegression()
ldang.fit(days, dang)

ldang.predict(np.array(10).reshape(-1,1))

lb.predict(np.array(10).reshape(-1,1))
