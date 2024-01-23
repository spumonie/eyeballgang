# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:49:11 2024

@author: lelep
"""

import packages.small_batch_images as sbim 
import pandas as pd
import numpy as np

df = sbim.load_annotation()
df_all = df
# I haven't tested if this code works in this environment. So far everything else works. 
# but i set it up so the evaluator can just ignore this if he wants to and use
# an already generated csv.

# Note, this data set takes a rather long time to generate, so we instead urge you to 
# just use the CSV file which is contained in 'outputs/RGB_mean_all_nosplit.csv'
# it is just the data set as generated according to the method below.
# if you want to do it, set the tag "generate_my_self" to True and check the 
# outputs file fore RGB_mean_all_nosplit.csv. It will be automatically used by 
# the model_generator file if it was generated. 
generate_my_self=False

if generate_my_self:
    #remove".head(20)" for generating entire dataset, this is just a test ammount
    dataset = pd.DataFrame(columns=['Age', 'LR', 'LG', 'LB', 'RR', 'RG', 'RB'])
    for i in range(len(df_all.head(20))):
        sample = df_all.head(20).iloc[i]
        sample = sample[:1] # pick ID and age only
    
        # store the info of the left eye
        img = sbim.load_left_eye_image(class_=None, df = df_all.head(20), i=i)
        img = np.array(img)
        img = sbim.cropToFundus2(img)
        # store the averaged number for each color channel
        colors =  ['R', 'G', 'B']
        for j in range(len(colors)):
            sample['L'+ colors[j]] = img[:, :, j].mean()
    
        # same for the right eye
        img = sbim.load_right_eye_image(class_=None, df = df_all.head(20), i=i)
        img = np.array(img)
        img = sbim.cropToFundus2(img)
        colors =  ['R', 'G', 'B']
        for j in range(len(colors)):
            sample['R'+ colors[j]] = img[:, :, j].mean()
    
        # add the current sample to the dataset
        dataset.loc[len(dataset)] = sample
    
    
        #dataset is now set up into distinct left and right eyes where one row contains:
            # 'Age', 'LR', 'LG', 'LB', 'RR', 'RG', 'RB' where: 
                # 'Age' is a patient's age,
                # 'LR', 'LG', 'LB', are the average R, G, B values for the left eye 
                # 'RR', 'RG', 'RB', are the average R, G, B values for the right eye 
        
        #for this submision, we are not distinguishing between left and right because
        #we understood that only one eye image will be fed to the model:
        temp_df1 = dataset[['Age','RR','RG','RB']].copy().rename(columns={'RR':'R','RG':'G','RB':'B'})
        temp_df2 = dataset[['Age','LR','LG','LB']].rename(columns={'LR':'R','LG':'G','LB':'B'}) 
        dataset_noLR = pd.concat([temp_df1, temp_df2], axis=0)
        
        #now each row looks like 'Age', 'R', 'G', 'B' and the data set has twice as 
        #many rows.
        
        dataset_noLR.to_csv('outputs/RGB_mean_all_nosplit_submission.csv')


    
