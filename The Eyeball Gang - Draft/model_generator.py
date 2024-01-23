# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:54:24 2024

@author: lelep
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# If you decided to generate the dataset yourself, then that is used. If not
# then an already provided csv file is used. They are equivalent. 
# our model is just a simple linear regression model trained on the Age as output
# and the (mean(R),mean(G),mean(B)) vector as input.

if os.path.exists("outputs/RGB_mean_all_nosplit_submission.csv"):
    dataset_noLR = pd.read_csv('outputs/RGB_mean_all_nosplit_submission.csv',index_col=0)
    dataset_noLR_numpy=dataset_noLR.to_numpy()

else:
    dataset_noLR = pd.read_csv('outputs/RGB_mean_all_nosplit.csv',index_col=0)
    dataset_noLR_numpy=dataset_noLR.to_numpy()


#train and test split
train, test = train_test_split(dataset_noLR_numpy)

# linear regression model
lr_model = LinearRegression()
lr_model.fit(train[:, 1:], train[:, 0])






