#%%
# Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# Reading in dataset
df = pd.read_csv('bank-additional-full.csv', sep=';', na_values='unknown')
df.head()
# %%
# Changing the name of columns to remove the periods
cols = df.columns
column = []
for col in cols:
    if '.' in col:
        words = col.split('.')
        actual_col = ''
        for i in range(len(words)):
            if i == len(words)-1:
                actual_col += words[i]
            else:
                actual_col += words[i]+'_'
        column.append(actual_col)
    else:
        column.append(col)
column
# %%
# Checking the number of null values in each columns
df.columns = column
df.isnull().sum()
# %%
# Exploring the behaviour of numerical features in the data
df.describe()
# %%
# Exporting the neater dataset into another dataset
df.to_csv('bank-marketing.csv', index=False)
# %%
