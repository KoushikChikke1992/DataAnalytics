#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Importing dataset and examining it
dataset = pd.read_csv("Patches.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


# In[3]:


#Converting cetegorical to numerical values
dataset['Tree'] = dataset['Tree'].map({'Other':0, 'Spruce':1})

print(dataset.info())


# In[4]:


# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')


# In[5]:


#Examining the dataset
print(dataset.info())


# In[15]:


# Dividing data into subsets
#Hydrology Data
subset1 = dataset[['Elevation','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Tree']]

#Roadways Data
subset2 = dataset[['Elevation','Slope','Horizontal_Distance_To_Roadways','Tree']]

#Fire_Points Data
subset3 = dataset[['Elevation','Slope','Horizontal_Distance_To_Fire_Points','Tree']]


# In[16]:


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X3 = feature_scaler.fit_transform(subset3)


# In[8]:


#Verifying if there are any NA values in the dataset
dataset.isna().sum()
dataset.isna()


# In[9]:


# Analysis on subset1 - Hydrology Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[10]:


# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X1)


# In[24]:


# Implementing t-SNE to visualize first subset
tsne = TSNE(n_components = 2, perplexity =30,n_iter=5000)
x_tsne = tsne.fit_transform(X1)

Elevation = list(dataset['Elevation'])
Slope = list(dataset['Slope'])
Horizontal_Distance_To_Hydrology = list(dataset['Horizontal_Distance_To_Hydrology'])
Vertical_Distance_To_Hydrology = list(dataset['Vertical_Distance_To_Hydrology'])
Tree = list(dataset['Tree'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Elevation: {a}; Slope: {b}; Horizontal_Distance_To_Hydrology:{c}, Vertical_Distance_To_Hydrology:{d}, Tree:{e}' for a,b,c,d,e in list(zip(Elevation,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')


# In[17]:


# Analysis on subset1 - Roadways Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[18]:


# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X2)


# In[20]:


# Implementing t-SNE to visualize the second subset
tsne = TSNE(n_components = 2, perplexity =30,n_iter=5000)
x_tsne = tsne.fit_transform(X2)

Elevation = list(dataset['Elevation'])
Slope = list(dataset['Slope'])
Horizontal_Distance_To_Roadways = list(dataset['Horizontal_Distance_To_Roadways'])
#Vertical_Distance_To_Hydrology = list(dataset['Vertical_Distance_To_Hydrology'])
Tree = list(dataset['Tree'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=subset2['Tree'], colorscale='Rainbow', opacity=0.5),
                                text=[f'Elevation: {a}; Slope: {b}; Horizontal_Distance_To_Roadways:{c}, Tree:{d}' for a,b,c,d in list(zip(Elevation,Slope,Horizontal_Distance_To_Roadways,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')


# In[21]:


# Analysis on subset1 - Fire_Points Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[22]:


# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X3)


# In[23]:


# Implementing t-SNE to visualize the third Subset
tsne = TSNE(n_components = 2, perplexity =30,n_iter=5000)
x_tsne = tsne.fit_transform(X3)

Elevation = list(dataset['Elevation'])
Slope = list(dataset['Slope'])
Horizontal_Distance_To_Fire_Points = list(dataset['Horizontal_Distance_To_Fire_Points'])
#Vertical_Distance_To_Hydrology = list(dataset['Vertical_Distance_To_Hydrology'])
Tree = list(dataset['Tree'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=subset3['Tree'], colorscale='Rainbow', opacity=0.5),
                                text=[f'Elevation: {a}; Slope: {b}; Horizontal_Distance_To_Fire_Points:{c}, Tree:{d}' for a,b,c,d in list(zip(Elevation,Slope,Horizontal_Distance_To_Fire_Points,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')


# In[ ]:




