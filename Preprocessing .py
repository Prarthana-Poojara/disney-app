#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


data = pd.read_csv("disney_movie_data_final_edit.csv")


# In[3]:


data.head()


# In[5]:


data.info()


# In[27]:


print(data.shape)


# In[31]:


movies_cleaned = data.drop(columns = ["Japanese","Hepburn","Adaptation by","Traditional","Simplified","Box office","Budget","Running time","Release date (datetime)","Production companies"])


# In[32]:


movies_cleaned.info()


# In[33]:


movies_cleaned.isnull().sum()


# Filling out NANs in object datatypes

# In[34]:


# Find the Columns which contain strings 
for label,content in movies_cleaned.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[35]:


movies_cleaned["Written by"] = movies_cleaned["Written by"].fillna("unknown")


# In[36]:


movies_cleaned["Production company"] = movies_cleaned["Production company"].fillna("unknown")
movies_cleaned["plot"] = movies_cleaned["plot"].fillna("unknown")
movies_cleaned["genre"] = movies_cleaned["genre"].fillna("unknown")
movies_cleaned["Based on"] = movies_cleaned["Based on"].fillna("unknown")
movies_cleaned["Screenplay by"] = movies_cleaned["Screenplay by"].fillna("unknown")
movies_cleaned["Country"] = movies_cleaned["Country"].fillna("unknown")


# In[37]:


movies_cleaned.isnull().sum()


# In[46]:


movies_cleaned["Release date"] = movies_cleaned["Release date"].fillna("unknown")
movies_cleaned["Language"] = movies_cleaned["Language"].fillna("unknown")
movies_cleaned["Directed by"] = movies_cleaned["Directed by"].fillna("unknown")
movies_cleaned["Produced by"] = movies_cleaned["Produced by"].fillna("unknown")
movies_cleaned["Starring"] = movies_cleaned["Starring"].fillna("unknown")
movies_cleaned["Music by"] = movies_cleaned["Music by"].fillna("unknown")
movies_cleaned["Distributed by"] = movies_cleaned["Distributed by"].fillna("unknown")
movies_cleaned["Story by"] = movies_cleaned["Story by"].fillna("unknown")
movies_cleaned["Narrated by"] = movies_cleaned["Narrated by"].fillna("unknown")
movies_cleaned["Cinematography"] = movies_cleaned["Cinematography"].fillna("unknown")
movies_cleaned["Screenplay by"] = movies_cleaned["Screenplay by"].fillna("unknown")
movies_cleaned["Edited by"] = movies_cleaned["Edited by"].fillna("unknown")
movies_cleaned["user_ratings"] = movies_cleaned["user_ratings"].fillna("unknown")
movies_cleaned["rotten_tmatoes"] = movies_cleaned["rotten_tmatoes"].fillna("unknown")


# In[47]:


movies_cleaned.isnull().sum()


# Filling out numerical missing values with median of other values 

# In[42]:


# check for which numeric columns have null values 
for label,content in movies_cleaned.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[44]:


# fill the numeric rows with the median 
for label,content in movies_cleaned.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # fill missing numeric value with median 
            movies_cleaned[label]=content.fillna(content.median())


# In[45]:


movies_cleaned.isnull().sum()


# In[48]:


movies_cleaned.head()


# In[49]:


movies_cleaned.to_csv("preprocessed_data.csv")


# In[ ]:




