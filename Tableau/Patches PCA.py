#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[14]:


#Retrieving the dataset and examining it
dataset = pd.read_csv("Patches.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


# In[15]:


#Checking for the datatypes of all the attributes in the dataset
dataset.dtypes


# In[16]:


#Checking for null values
dataset.isnull().sum()


# In[17]:


#Dropping the NA values
dataset=dataset.dropna()
dataset=dataset.mask(dataset.eq('BLANK')).dropna()
dataset=dataset.mask(dataset.eq(' ')).dropna()


# In[18]:


# Converting Categorical features into Numerical features
converter = LabelEncoder()
dataset['Tree'] = converter.fit_transform(dataset['Tree'].astype(str))


# In[20]:


# Dividing dataset into label and feature sets
X = dataset.drop(['Tree'], axis = 1) # Features
Y = dataset['Tree'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)


# In[21]:


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# In[10]:


# Implementing PCA to visualize dataset
pca = PCA(n_components = 2)
pca.fit(X_scaled)
x_pca = pca.transform(X_scaled)
print("Variance explained by each of the n_components: ",pca.explained_variance_ratio_)
print("Total variance explained by the n_components: ",sum(pca.explained_variance_ratio_))
plt.figure(figsize = (8,6))
plt.scatter(x_pca[:,0], x_pca[:,1],c=Y, cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Dimensionality Reduction')
plt.show()


# In[12]:


# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity = 30)
x_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize = (8,6))
plt.scatter(x_tsne[:,0], x_tsne[:,1],c=Y, cmap='plasma')
plt.xlabel('First Dimension')
plt.ylabel('Second Dimension')
plt.title('t-SNE Dimensionality Reduction')
plt.show()


# In[ ]:




