#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


# In[4]:


import os
data_path = os.path.abspath('C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images')


# In[5]:


vgg = Xception(input_shape=[224,224,3], weights='imagenet', include_top=False)


# In[6]:


# For not training the VGG weights
for layer in vgg.layers:
  layer.trainable = False


# In[7]:


x = Flatten()(vgg.output)
prediction = Dense(5, activation='softmax')(x)


# In[8]:


model = Model(inputs=vgg.input, outputs=prediction)


# In[9]:


model.summary()


# In[10]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy'])


# In[11]:


datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function = preprocess_input)


# In[12]:


train_generator = datagen.flow_from_directory(
    data_path,
    subset='training',
    target_size=[224,224],
    classes = ['Mild','No_DR','Moderate','Proliferate','Severe'],
    class_mode = 'categorical')


# In[13]:


valid_generator = datagen.flow_from_directory(
    data_path, 
    subset='validation',
    target_size=[224,224],
    classes = ['Mild','No_DR','Moderate','Proliferate','Severe'],
    class_mode = 'categorical')


# In[14]:


r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=12,
  steps_per_epoch= 1,
  verbose = 1)


# In[15]:


plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[16]:


model.save("XceptionModel.h5")


# In[17]:


from keras import models
model = models.load_model('XceptionModel.h5')


# In[18]:


Y_pred = model.predict_generator(valid_generator)


# In[19]:


predicted = []
for x in Y_pred:
    predicted.append(np.where(x == max(x))[0])


# In[20]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(valid_generator.classes,predicted)


# In[22]:


import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[23]:


plot_confusion_matrix(cm           = cm, 
                      normalize    = True,
                      target_names = ['Mild','No_DR','Moderate','Proliferate','Severe'],
                      title        = "Confusion Matrix, Normalized")


# In[ ]:




