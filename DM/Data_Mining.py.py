#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import seaborn as sns


# In[2]:


dataset = pd.read_csv("Employee.csv", encoding = "ISO-8859-1")


# In[3]:


#Initial Data Exploration

dataset.shape


# In[4]:



dataset.shape


# In[5]:


dataset.describe()


# In[8]:


dataset.head(5)


# In[6]:


dataset.isnull().any()


# In[7]:


#heatmap
import matplotlib.pyplot as plt
import seaborn as sns
correlations_sub = dataset.corr()
plt.figure(figsize=(26,26))
sns.heatmap(correlations_sub, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[8]:


#scatterplot
####sns.pairplot(dataset, size = 2.5)
###plt.show();


# In[9]:


#Checking the class distribution of target variable
sns.countplot(x='Attrition', data=dataset).set_title('Distribution of Target variable "Attrition"')
plt.show()


# In[10]:


dataset.hist(bins=30,figsize=(18,18))
plt.show()


# In[12]:


skew_data=dataset.skew()
print(skew_data)


# In[14]:


#Checking the class distribution of Independent variable
sns.countplot( x='BusinessTravel',data=dataset).set_title('Distribution of Business Travel')
plt.show()


# In[15]:


#Checking the class distribution of Independent variable
sns.countplot( x='Department',data=dataset).set_title('Department')
plt.show()


# In[16]:


#Checking the class distribution of Independent variable
sns.countplot( x='Education',data=dataset).set_title('Education')
plt.show()


# In[17]:


#Checking the class distribution of Independent variable
sns.countplot( x='Gender',data=dataset).set_title('Gender')
plt.show()


# In[18]:


#Checking the class distribution of Independent variable
sns.countplot( x='JobRole',data=dataset).set_title('JobRole')
plt.show()


# In[19]:


#Checking the class distribution of Independent variable
sns.countplot( x='MaritalStatus',data=dataset).set_title('MaritalStatus')
plt.show()


# In[20]:


#Checking the class distribution of Independent variable
sns.countplot( x='Over18',data=dataset).set_title('Over18')
plt.show()


# In[21]:


#Checking the class distribution of Independent variable
sns.countplot( x='OverTime',data=dataset).set_title('OverTime')
plt.show()


# In[ ]:




