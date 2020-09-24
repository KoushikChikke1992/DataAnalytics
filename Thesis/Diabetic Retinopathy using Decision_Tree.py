#!/usr/bin/env python
# coding: utf-8

# In[8]:


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import tree
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix


# In[2]:


def load_image_files(container_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


# In[3]:


image_dataset = load_image_files("C:/Users/Input/gaussian_filtered_images/gaussian_filtered_images")


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)


# In[5]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# In[6]:


y_pred = clf.predict(X_test)


# In[7]:


print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))


# In[9]:


confusion_matrix(y_test, y_pred)


# In[ ]:


confusion_matrix(y_test, y_pred)

