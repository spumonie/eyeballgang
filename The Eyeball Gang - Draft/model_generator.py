# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:54:24 2024

@author: lelep
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Following code is in case that evaluator actually wants to generate our 
# dataset like we did. It selects the file genereated in dataset generator if he
# generated it. Doesn't work unless I can find a proper substition for the 
# first if statment: "if outputs/RGB_mean_all_nosplit_submission.csv exists:"
"""
if outputs/RGB_mean_all_nosplit_submission.csv exists: #need code that says this
    dataset_noLR = pd.read_csv('outputs/RGB_mean_all_nosplit_submission.csv',index_col=0)
    dataset_noLR_numpy=dataset_noLR.to_numpy()

else:
    dataset_noLR = pd.read_csv('outputs/RGB_mean_all_nosplit.csv',index_col=0)
    dataset_noLR_numpy=dataset_noLR.to_numpy()
"""

# this can be replaced with the above if/else statement once it's fixed.
dataset_noLR = pd.read_csv('outputs/RGB_mean_all_nosplit.csv',index_col=0)
dataset_noLR_numpy=dataset_noLR.to_numpy()

#train and test split
train, test = train_test_split(dataset_noLR_numpy)

# linear regression model
lr_model = LinearRegression()
lr_model.fit(train[:, 1:], train[:, 0])

# for model evaluation see jupyter notebook titled "name here" thought maybe 
# to include for model information for the reader. 




