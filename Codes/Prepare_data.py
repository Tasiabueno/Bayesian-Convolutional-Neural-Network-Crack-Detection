# Dowload the dataset from "https://data.mendeley.com/datasets/xnzhj3x8v4/2"

# IMPORT THE FOLLOWING PACKAGES TO RUN THE CODE CORRECTLY 
import numpy as np
import pandas as pd
from random import shuffle
import os
import random
import cv2 

CATEGORIES = ["Negative", "Positive"] 

def shuffle_data(directory,DP):
    for category in CATEGORIES:
        path = os.path.join(directory, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array =(os.path.join(path,img))
            DP.append([img_array,class_num,category])
    random.shuffle(DP)

def create_data(files,X,y,label):
    for i in range(len(files)):
        y.append(files[i][1])
        label.append(files[i][2])
        img_array= cv2.imread(files[i][0], 0)
        img_array_resized = cv2.resize(img_array, (100,100))
        X.append(img_array_resized)
