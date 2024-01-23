# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:23:53 2024

@author: lelep
"""

import numpy as np
import packages.small_batch_images as sbim
import model_generator
from PIL import Image

# we assume here that you are just going to evaluate what we have done here
# like in the hackathon meeting, namely by feeding a single image into 
# our pipeline and seeing if it spits out an age
if __name__ == "__main__":
    
    #if testing single image just, place test image in same directory as this file
    path_to_image="0_left.jpg" #using 0_left.jpg here
    img = Image.open(path_to_image)
    input_array=sbim.singleImageProcessor(img) 
    result=np.round(model_generator.lr_model.predict(input_array))
    print(int(result))






