# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:10:45 2019

@author: sharm
"""

import pandas as pd

#read csv file and convert it in data frame

df=pd.read_csv("Salaries.csv")

# list first 5 records(Default)
df.head()
#list first 10 records
df.head(10)

#list last 5 records(default)
df.tail()

#list lat 10 records
df.tail(10)

#check type of all columns
df.dtypes
# to find all columns. This returns index
df.columns

#list the rown and column labels. It returns an object containg info about x axis and y axis
df.axes
# list number of dimensions in data frame
df.ndim

# size of elements
df.size

# to find dimensionality(structure) of data frame. It returns  tuple (rows,column)
df.shape

#numpy representation of data. It returns an object that contains lists
df.values

#This gives complete csv data
df.describe
# gives descriptive statistics(for numeric column only)
df.describe()
# to find max and min value. It works on both numerical and categorial data both
df.max()

df.min()

# find mean and works only one numeric data
df.mean()
df.median()
df.std()
df.mode()

# returns a random sample of the data frame
df.sample(10)  

# to find mean value of first 50 records
df.head(50).mean()

"""Selecting a column in a Data Frame
Method 1: Subset the data frame using column name:
df['sex']
Method 2: Use the column name as an attribute:
df.sex
Note:there is an attribute rankfor pandas data frames, 
so to select a column with a name "rank" we should use method 1.
"""


df['rank']
df.phd

#apply basic statistics on salary 
df['salary'].mean()
df['salary'].describe()
# to find number of countin column salary
df['salary'].count()
df.count()

"""
Data Frames groupbymethod
26
Using "group by" method we can:
•Split the data into groups based on some criteria
•Calculate statistics (or apply a function) to each group
•Similar to dplyr() function in R

"""

# group data by rank
df_rank=df.groupby(['rank'])
# calculate mean value for each numeric column per group
df_rank.mean()

df_rank.describe()

# find mean salary for each gropu of rank(profeesor)
a=df.groupby(['rank'])[['salary','service']].mean()

# or use
b=df.groupby(['rank'])[['salary']].mean()

"""
Important
Note:If single brackets are used to specify the column (e.g. salary), 
then the output is Pandas Series object. 
When double brackets are used the output is a Data Frame
"""
df.groupby('rank')[["salary"]].mean()

"""

groupbyperformance notes:
-no grouping/splitting occurs until it's needed. 
Creating the groupby object only verifies that you have passed 
a valid mapping
-by default the group keys are sorted during the 
groupby operation. You may want to pass sort=False 
for potential speedup:
"""
#Calculate mean salary for each professor rank:

df.groupby(['rank'],sort=True)[['salary']].mean()

"""
Data Frame: filtering
29
To subset the data we can apply Boolean indexing. 
This indexing is commonly known as a filter. 
For example if we want to subset the rows in which the salary
 value is greater than $120K:
"""
#Calculate mean salary for each professor rank:
df_sub=df[df['salary']>120000]

# select inly those rows that contains only female professor

df_f=df[df['sex']=="Female"]

"""
Data Frames: Slicing

When selecting one column, it is possible to use single set of 
brackets, but the resulting object will be a Series (not a DataFrame)


When we need to select more than one column and/or make the 
output to be a DataFrame, we should use double brackets

"""

#Select column salary:
dd=df['salary']

dd2=df[["salary","sex"]]

"""
Data Frames: method loc
33
If we need to select a range of rows, using their labels 
we can use method loc
"""
df.loc[10:20,['rank','sex','salary']]

"""
Data Frames: method iloc

If we need to select a range of rows and/or columns, 
using their positions we can use method iloc
"""

df.iloc[10:20,[0,3,4,5]]

df.iloc[0]
df.iloc[-1]
df.iloc[:,3]
df.iloc[:,-1]
df.iloc[0:7]
df.iloc[:,0:2]
df.iloc[1:3,0:2]
df.iloc[[0,5],[1,3]]# 0 and 5th row and 1st and 3rd column

"""
DataFrame sorting
"""

# Create a new data frame from the original sorted by the column Salary

df_sorted=df.sort_values(by='service')
df_sorted.head(10)

#We can sort the data using 2 or more columns:
df_sorted=df.sort_values(by=['service','salary'],ascending=[True,True])
df_sorted.head(10)

"""
Missing Values

Missing values are marked as NaN
"""
import pandas as pd
# Read a dataset with missing values
salary=pd.read_csv("Salaries.csv")
salary.count()
salary[salary.isnull().any(axis=1)].head() #select row that have atleast one missing value

salary[salary.isnull().all(axis=1)]#select the rows with all values NaN

salary.dropna(axis=0,how='any')

"""
dropna(how='all') >> Drop observations where all cells is NA
dropna(axis=1, how='all') >> Drop column if all the values aremissing
dropna(thresh = 5) >> Drop rows that contain less than 5 non-missing values
fillna(0) >> Replace missing values with zeros
isnull() >> returns True if the value is missing
notnull() >> Returns True for non-missing values

More notes:
    Missing Values

•When summing the data, missing values will be treated as zero
•If all values are missing, the sum will be equal to NaN
•cumsum() and cumprod() methods ignore missing values but 
preserve them in the resulting arrays
•Missing values in GroupBymethod are excluded (just like in R)
•Many descriptive statistics methods have skipnaoption to control 
if missing data should be excluded . This value is set to True by 
default (unlike R)

"""

# fill the record having missing values, with mean of that column
salary['salary']=salary['salary'].fillna(salary['salary'].mean())
salary[salary.isnull().any(axis=1)]

# fill all the records having missing values, with mean of that column
salary = salary.fillna(salary.mean())
salary[salary.isnull().any(axis=1)]

#agg() method are useful when multiple statistics are computed per column:
salary[['service','salary']].agg(['min','mean','max'])




