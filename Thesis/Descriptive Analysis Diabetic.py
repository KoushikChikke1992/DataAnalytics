#!/usr/bin/env python
# coding: utf-8

# Necessory to import required libraries
# 
# pandas
# numpy
# matplotlib
# keras
# sklearn
# PIL and so on

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from PIL import Image
from glob import glob
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense , Dropout, Flatten , Conv2D,MaxPool2D
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('C:/Users/Input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# mage to Array conversion and converting the data typr to unit8 to show the images using matplotlib

# In[2]:


def read_images(path,num_img):
    array=np.zeros((num_img,224,224,3))
    i=0
    for img in os.listdir(path):
        img_path=path + "/" + img
        img=Image.open(img_path,mode="r")
        data=np.asarray(img,dtype="uint8")
        array[i]=data
        i+=1
    return array


# In[3]:


no_dr_path=r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/No_DR"
num_no_dr=len(glob("C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/No_DR/*"))
no_dr_array=read_images(no_dr_path,num_no_dr)
no_dr_array=no_dr_array.astype(np.uint8)

mild_path=r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Mild"
num_mild=len(glob("C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Mild/*"))
mild_array=read_images(mild_path,num_mild)
mild_array=mild_array.astype(np.uint8)

moderate_path=r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Moderate"
num_moderate=len(glob("C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Moderate/*"))
moderate_array=read_images(moderate_path,num_moderate)
moderate_array=moderate_array.astype(np.uint8)

proliferate_dr_path=r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Proliferate_DR"
num_proliferate_dr=len(glob("C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Proliferate_DR/*"))
proliferate_dr_array=read_images(proliferate_dr_path,num_proliferate_dr)
proliferate_dr_array=proliferate_dr_array.astype(np.uint8)

severe_path =r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Severe"
num_severe=len(glob("C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Severe/*"))
severe_array=read_images(severe_path,num_severe)
severe_array=severe_array.astype(np.uint8)


# shape of the learning images

# In[4]:


print("no_dr_array",no_dr_array.shape)
print("mild_array",mild_array.shape)
print("moderate_array",moderate_array.shape)
print("proliferate_dr_array",proliferate_dr_array.shape)
print("severe_array",severe_array.shape)


# Sample image

# In[5]:


plt.imshow(no_dr_array[5])
plt.axis("off")
plt.show()


# data type of the image array

# In[6]:


print("no_dr_array dtype is",no_dr_array.dtype)
print("mild_array dtype is",mild_array.dtype)
print("moderate_array dtype is",moderate_array.dtype)
print("proliferate_dr_array dtype is",proliferate_dr_array.dtype)
print("severe_array dtype is",severe_array.dtype)


# Number of classes(Labels) and samples to be traind on NO DR Mild Moderate Proliferate severe

# In[7]:


print("num_no_dr:",num_no_dr)
print("num_mild:",num_mild)
print("num_moderate:",num_moderate)
print("num_proliferate_dr:",num_proliferate_dr)
print("num_severe:",num_severe)


# labeling the samples : No DR with 0 Mild with 1 Moderate with 2 Proliferate with 3 Severe with 4

# In[9]:


zeros=np.zeros(1805)
ones=np.ones(370)
twos=np.full(999,2)
threes=np.full(295,3)
fours=np.full(193,4)
y = np.concatenate((zeros,ones,twos,threes,fours),axis=0)
print("y shape",y.shape)


# In[10]:


sns.countplot(y)
plt.show()


# image resizing to 64 * 64 * 3 for fast learning as we increase the image size the model train and predict more promising result as the the feature increses

# In[11]:


def resize_images(img):
    number_of_image=img.shape[0]
    new_array=np.zeros((number_of_image,64,64,3))
    for i in range(number_of_image):
        new_array[i]=cv2.resize(img[i,:,:,:],(64,64))
    return new_array
no_dr_array=resize_images(no_dr_array)
no_dr_array=no_dr_array.astype(np.uint8)
mild_array=resize_images(mild_array)
mild_array=mild_array.astype(np.uint8)
moderate_array=resize_images(moderate_array)
moderate_array=moderate_array.astype(np.uint8)
proliferate_dr_array=resize_images(proliferate_dr_array)
proliferate_dr_array=proliferate_dr_array.astype(np.uint8)
severe_array=resize_images(severe_array)
severe_array=severe_array.astype(np.uint8)


# In[12]:


plt.imshow(no_dr_array[5])
plt.axis("off")
plt.show()


# shape of the newly generated array

# In[13]:


print("new no_dr_array shape",no_dr_array.shape)
print("new mild_array shape",mild_array.shape)
print("new moderate_array shape",moderate_array.shape)
print("new proliferate_dr_array shape",proliferate_dr_array.shape)
print("new severe_array shape",severe_array.shape)


# In[ ]:




