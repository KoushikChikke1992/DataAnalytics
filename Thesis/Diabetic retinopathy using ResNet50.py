#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50

from glob import glob


# In[2]:


import os
data_path = os.path.abspath('C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images')


# In[3]:


resNet = ResNet50(input_shape=[224,224,3], weights='imagenet', include_top=False)


# In[4]:


# For not training the VGG weights
for layer in resNet.layers:
  layer.trainable = False


# In[5]:


x = Flatten()(resNet.output)
prediction = Dense(5, activation='softmax')(x)


# In[6]:


model = Model(inputs=resNet.input, outputs=prediction)


# In[7]:


model.summary()


# In[8]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy'])


# In[9]:


datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function = preprocess_input)


# In[10]:


train_generator = datagen.flow_from_directory(
    data_path,
    subset='training',
    target_size=[224,224],
    classes = ['Mild','No_DR','Moderate','Proliferate','Severe'],
    class_mode = 'categorical')


# In[11]:


valid_generator = datagen.flow_from_directory(
    data_path, 
    subset='validation',
    target_size=[224,224],
    classes = ['Mild','No_DR','Moderate','Proliferate','Severe'],
    class_mode = 'categorical')


# In[12]:


r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=12,
  steps_per_epoch= 1,
  verbose = 1)


# In[13]:


plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




