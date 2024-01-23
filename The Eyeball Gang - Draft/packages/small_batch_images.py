# Helping code for our modules
import pandas as pd
from PIL import Image
import os
import numpy as np

script_dir = os.path.dirname(__file__)
classes = ["N", "D", "G", "C", "A", "H", "M", "O"]
classes_name = ["normal", "diabetes", "glaucoma", "cataract", "AMD", "hypertensi\
    on", "myopia", "other diseases"]

# loads the non-image data
def load_annotation():
    df = pd.read_csv(os.path.join(script_dir, "../annotations.csv"), index_col=0)
    return df

# loads left eye images
def load_left_eye_image(df, i, class_=None ):
    if class_==None:
        return Image.open(os.path.join(script_dir, "../images/{}").format(df.Left_Fundus.values[i]))
    image = Image.open(os.path.join(script_dir, "../images/{}").format(df[df[class_] == 1].Left_Fundus.values[i]))
    return image

#loads right eye images
def load_right_eye_image(df,i, class_=None):
    if class_==None:
        return Image.open(os.path.join(script_dir, "../images/{}").format(df.Right_Fundus.values[i]))
    image = Image.open(os.path.join(script_dir, "../images/{}").format(df[df[class_] == 1].Right_Fundus.values[i]))
    return image 

#For working with Numpyarrays to crop image down to the eye itself (minimise black background)   
def cropToFundus2(x):
    max_ = x.max(axis=2)
    valid_columns = np.where(max_.max(axis=0) > 30)[0]
    valid_rows = np.where(max_.max(axis=1) > 30)[0]
    x = x[valid_rows, : , :]
    x = x[:, valid_columns, :]
    return x

# Similar to CropToFundus2 but with added line to take opened image as input
# and lines to return rbg_vector of image with proper shape for the LR model
def singleImageProcessor(x):
    x = np.array(x)
    max_ = x.max(axis=2)
    valid_columns = np.where(max_.max(axis=0) > 30)[0]
    valid_rows = np.where(max_.max(axis=1) > 30)[0]
    x = x[valid_rows, : , :]
    x = x[:, valid_columns, :]
    rbg_vector= np.array([x[:,:,0].mean(),x[:,:,1].mean(),x[:,:,2].mean()])
    rbg_vector = np.reshape(rbg_vector, (1, -1))
    return rbg_vector
