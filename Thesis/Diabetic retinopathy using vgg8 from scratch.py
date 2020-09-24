#!/usr/bin/env python
# coding: utf-8

# # Diabetic retinopathy using vgg8

# For building Keras model necessory to import required libraries
# 
# 1. pandas
# 2. numpy
# 3. matplotlib
# 4. keras
# 5. sklearn
# 6. PIL and so on

# In[1]:


import tensorflow as tf
import keras
import numpy as np
import matplotlib
import sklearn
import cv2
import os
import glob
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, KFold
from keras import models
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Flatten, Dropout
from keras.optimizers import Adam, SGD
from PIL import Image
from keras.utils import to_categorical


# Procedure to read images from the folders according to specified path and number of images

# In[2]:


def read_images(path, number_of_images):
    arr = np.zeros((number_of_images, 224, 224, 3))
    i = 0
    for image in os.listdir(path): #image will be the name of the file
        image_path = path + "/" + image #creating a full path for the image 
        image = Image.open(image_path, mode='r')
        image_data = np.asarray(image, dtype='uint8')
        arr[i] = image_data
        i += 1
    return arr


# Reading the images from different paths according to provided catogories and converting tham as unit8 images
# 
# 1. NO DR
# 2. Mild
# 3. Moderate
# 4. Proliferate
# 5. Severe

# In[3]:


def read_images_in_path(category):
    if category == 0: #No_DR
        path = r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/No_DR"
    elif category == 1: #Mild
        path = r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Mild"
    elif category == 2: #Moderate
        path = r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Moderate"
    elif category == 3: #Proliferate_DR
        path = r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Proliferate_DR"
    elif category == 4: #Severe
        path = r"C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images/Severe"
    else:
        raise ValueError('Invalid category')
    end_path = path + '/*'
    num_in_path = len(glob.glob(end_path))
    images = read_images(path, num_in_path)
    images = images.astype('uint8')
    return num_in_path, images


# In[4]:


def read_images1(path, number_of_images):
    arr = np.zeros((number_of_images, 224, 224,3))
    i = 0
    for image in os.listdir(path): #image will be the name of the file
        image_path = path + "/" + image #creating a full path for the image 
        image = Image.open(image_path, mode='r')
        image_data = np.asarray(image, dtype='uint8')
        arr[i] = image_data
        i += 1
    return arr


# In[5]:


def read_images_in_pathGray(category):
    if category == 0: #No_DR
        path = r"C:/Users/Downloads/Compressed/Diabetic Retino grayscale/grayscale_images/grayscale_images/No_DR"
    elif category == 1: #Mild
        path = r"C:/Users/Downloads/Compressed/Diabetic Retino grayscale/grayscale_images/grayscale_images/Mild"
    elif category == 2: #Moderate
        path = r"C:/Users/Downloads/Compressed/Diabetic Retino grayscale/grayscale_images/grayscale_images/Moderate"
    elif category == 3: #Proliferate_DR
        path = r"C:/Users/Downloads/Compressed/Diabetic Retino grayscale/grayscale_images/grayscale_images/Proliferate_DR"
    elif category == 4: #Severe
        path = r"C:/Users/Downloads/Compressed/Diabetic Retino grayscale/grayscale_images/grayscale_images/Severe"
    else:
        raise ValueError('Invalid category')
    end_path = path + '/*'
    num_in_path = len(glob.glob(end_path))
    images = read_images1(path, num_in_path)
    images = images.astype('uint8')
    return num_in_path, images


# Pixals normalization for quick and accurate training by dividing images with 255

# In[6]:


def normalize_pixels(images):
    images = images.astype('float32')
    images = images/255
    return images


# Decreasing the resolution of the images

# In[7]:


def decrease_res(images, num_images, res):
    new_images = np.zeros((num_images, res, res, 3))
    i = 0
    for image in images:
        new_image     = cv2.resize(image, (res,res))
        new_images[i] = new_image
        i += 1
    return new_images


# reading images according toits catogories 

# In[8]:


No_DR_num, No_DR_images   = read_images_in_path(0) #1805
Mild_num, Mild_images     = read_images_in_path(1) #370
Mod_num, Mod_images       = read_images_in_path(2) #999
Prolif_num, Prolif_images = read_images_in_path(3) #295
Severe_num, Severe_images = read_images_in_path(4) #193


# Sample image

# In[9]:


pyplot.imshow(Mild_images[20])
pyplot.axis("off")
pyplot.show()


# nomalizing the images for faster learning

# In[10]:


No_DR_images  = normalize_pixels(No_DR_images)
Mild_images   = normalize_pixels(Mild_images)
Mod_images    = normalize_pixels(Mod_images)
Prolif_images = normalize_pixels(Prolif_images)
Severe_images = normalize_pixels(Severe_images)


# We now generate a corresponding label vector that will match images to a category.

# In[11]:


no_DR  = np.zeros(No_DR_num)
mild   = np.ones(Mild_num)
mod    = np.full(Mod_num, 2)
prolif = np.full(Prolif_num, 3)
severe = np.full(Severe_num, 4)
labels = np.concatenate((no_DR, mild, mod, prolif, severe), axis=0)
labels = to_categorical(labels)


# one to one corresponding of labels with images 

# In[12]:


x = np.concatenate((No_DR_images, Mild_images, Mod_images, Prolif_images, Severe_images))
y = labels


# Splitting the data into traingset and testset for model training 20% test data remaing 
# 80% is consist of 20% validation and rest for training

# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.2, random_state=42)


# decreasing the resolution to 64 * 64

# In[14]:


x_train  = decrease_res(x_train, np.shape(x_train)[0], res=64)
x_val    = decrease_res(x_val, np.shape(x_val)[0], res=64)
x_test   = decrease_res(x_test, np.shape(x_test)[0], res=64)


# In[ ]:





# Shape of the training, validation and test data 

# In[15]:


print(np.shape(x_train))
print(np.shape(x_val))
print(np.shape(x_test))


# Let's get around to defining the VGG8 model, whose architecture is like a compact version of the VGG16 model.

# In[16]:


def define_VGG8():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=(64,64,3)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation = 'softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


# Model evaluation function

# In[17]:


def evaluate_model(trainX, trainY, valX, valY, model, batch_size, epochs):
    hist = model.fit(trainX, trainY, batch_size, epochs, verbose=1, validation_data=(valX, valY))
    _, train_score = model.evaluate(trainX,trainY)
    _, val_score  = model.evaluate(valX,valY)
    return hist, train_score, val_score


# Summarizing results of a particular model

# In[18]:


def results_summary(hist):
    pyplot.subplot(2,1,1)
    pyplot.title('Loss')
    pyplot.plot(hist.history['loss'], color='blue',label='Train')
    pyplot.plot(hist.history['val_loss'], color='orange', label='Validation')
    pyplot.subplot(2,1,2)
    pyplot.title('Accuracy')
    pyplot.plot(hist.history['accuracy'], color='blue', label='Train')
    pyplot.plot(hist.history['val_accuracy'], color='orange', label = 'Validation')
    pyplot.show()


# Model fitting and training

# In[19]:


model   = define_VGG8()
hist, train_score, val_score = evaluate_model(x_train, y_train, x_val, y_val, model, batch_size=32, epochs=75)


# In[20]:


results_summary(hist)


# Clearly, the model is overfitting, as the validation loss continues to go up. However, the validation acucuracy is fairly stable. We should try changing hyperparameters such decreasing batch size and epochs.

# In[21]:


model2 = define_VGG8()
model3 = define_VGG8()
model4 = define_VGG8()

hist2, train_score2, val_score2 = evaluate_model(x_train, y_train, x_val, y_val, model2, batch_size=32, epochs=30)
hist3, train_score3, val_score3 = evaluate_model(x_train, y_train, x_val, y_val, model3, batch_size=16, epochs=30)
hist4, train_score4, val_score4 = evaluate_model(x_train, y_train, x_val, y_val, model4, batch_size= 8, epochs=50)


# In[22]:


results_summary(hist2)
results_summary(hist3)
results_summary(hist4)


# Out of the models, Model 4 seems to perform the best. It doesn't seem to overfit and achieves the best performance. Let's use these specifications to create the final model.

# In[23]:


def fit_model(x_train, y_train, model, batch_size, epochs):
    history = model.fit(x_train, y_train, batch_size, epochs, verbose=1)
    _, train_score = model.evaluate(x_train, y_train)
    model.save('Diabetic_Retinopathy_Model.h5')
    return history, train_score


# In[24]:


def fit_model1(x_train, y_train, model, batch_size, epochs):
    history = model.fit(x_train, y_train, batch_size, epochs, verbose=1)
    _, train_score = model.evaluate(x_train, y_train)
    model.save('Diabetic_Retinopathy_Model2.h5')
    return history, train_score


# In[25]:


final_model = define_VGG8()
hist, train_score = fit_model(x_train, y_train, final_model, batch_size=8, epochs=50)


# For validation we can test the mode on test data

# In[26]:


final_model = models.load_model('Diabetic_Retinopathy_Model.h5')
_, score = final_model.evaluate(x_test, y_test)
print(score)


# the model shows 70% accuracy on test set

# Perparing data for transfer learning 
# reading images according toits catogories

# In[27]:


No_DR_num1, No_DR_images1   = read_images_in_pathGray(0) #1805
Mild_num1, Mild_images1     = read_images_in_pathGray(1) #370
Mod_num1, Mod_images1       = read_images_in_pathGray(2) #999
Prolif_num1, Prolif_images1 = read_images_in_pathGray(3) #295
Severe_num1, Severe_images1 = read_images_in_pathGray(4) #193


# sample image in retraining model

# In[28]:


pyplot.imshow(Mild_images1[15])
pyplot.axis("off")
pyplot.show()


# nomalizing the images for faster learning

# In[29]:


No_DR_images1  = normalize_pixels(No_DR_images1)
Mild_images1   = normalize_pixels(Mild_images1)
Mod_images1    = normalize_pixels(Mod_images1)
Prolif_images1 = normalize_pixels(Prolif_images1)
Severe_images1 = normalize_pixels(Severe_images1)


# We now generate a corresponding label vector that will match images to a category.

# In[30]:


no_DR1  = np.zeros(No_DR_num1)
mild1   = np.ones(Mild_num1)
mod1    = np.full(Mod_num1, 2)
prolif1 = np.full(Prolif_num1, 3)
severe1 = np.full(Severe_num1, 4)
labels1 = np.concatenate((no_DR1, mild1, mod1, prolif1, severe1), axis=0)
labels1 = to_categorical(labels1)


# In[31]:


x1 = np.concatenate((No_DR_images1, Mild_images1, Mod_images1, Prolif_images1, Severe_images1))
y1 = labels1


# In[32]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(x1,y1, test_size=0.2, random_state=42)
x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train1,y_train1, test_size=0.2, random_state=42)


# In[33]:


x_train1  = decrease_res(x_train1, np.shape(x_train1)[0], res=64)
x_val1    = decrease_res(x_val1, np.shape(x_val1)[0], res=64)
x_test1   = decrease_res(x_test1, np.shape(x_test1)[0], res=64)


# shape of the data

# In[34]:


print(np.shape(x_train1))
print(np.shape(x_val1))
print(np.shape(x_test1))


# Loading the model and training it on train data 

# In[37]:


final_model1 = models.load_model('Diabetic_Retinopathy_Model.h5')
hist1, train_score1 = fit_model1(x_train1, y_train1, final_model1, batch_size=8, epochs=50)


# In[40]:


final_model1 = models.load_model('Diabetic_Retinopathy_Model2.h5')
histogram, score1 = final_model.evaluate(x_test1, y_test1)
print(score1)


# We achived almost 75% accuracy except the fact tha training data was very less but still we achived good accuracy  

# In[ ]:




