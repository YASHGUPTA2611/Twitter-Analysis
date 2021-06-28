#!/usr/bin/env python
# coding: utf-8

# # A machine learning model has been made to identify when an article might be fake news

# ## Importing libraries which will be in use

# In[1]:


import nltk
import pandas as pd
import numpy as np
import tensorflow as tf


# In[2]:


# Reading training dataset
train = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\fake news\train.csv')


# In[3]:


# Top 5 rows
train.head()


# In[4]:


#total rows and columns
train.shape


# In[5]:


# Null values in the dataset
train.isnull().sum()


# In[6]:


# Dropping all the null values
train.dropna(inplace=True)


# In[7]:


# Again checking null values
train.isnull().sum()


# In[8]:


# Making a copy of train dataset and storing in message 
message = train.copy()


# In[9]:


# Resetting index because we have remove null values so the indexes are not alinged  in order. 
message.reset_index(inplace=True)


# In[10]:


message.head(15)


# In[11]:


message['title'][5]


# In[12]:


message.shape


# In[13]:


# Storing values of text column in col variable
col = message['text']


# ## Stemming on train data

# In[14]:


import re


# In[15]:


# importing nltk objects which will be in further use.
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps =  PorterStemmer()


# In[ ]:


# using for loop to apply stemming for every word.
corpus=[]
for i in range(len(message)):
    review= re.sub('[^a-zA-Z]', ' ', col[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)


# In[ ]:


corpus


# In[ ]:


corpus[5]


# In[ ]:


len(corpus)


# ## TF-IDF on train data
# 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray() 


# In[ ]:


X.shape


# In[ ]:


X


# In[ ]:


len(tfidf.get_feature_names())


# In[ ]:


# Storing values of output variables in y.
y = train['label'].values


# ## Applying Naive Bayes Model

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


clf = GaussianNB()


# In[ ]:


fake_news = clf.fit(X, y)


# ### Applying Tf-Idf vectorizor on our test dataset

# In[ ]:


test = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\fake news\test.csv')


# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test.dropna(inplace=True)


# In[ ]:


message_test = test.copy()


# In[ ]:


message_test.reset_index(inplace=True)


# In[ ]:


col2 = message_test['text']


# In[ ]:


col2[6]


# In[ ]:


corpus2=[]
for i in range(len(message_test)):
    reviewe= re.sub('[^a-zA-Z]', ' ', col2[i])
    reviewe = reviewe.lower()
    reviewe = reviewe.split()
    reviewe = [ps.stem(word) for word in reviewe if word not in set(stopwords.words('english'))]
    reviewe= ' '.join(reviewe)
    corpus2.append(reviewe)


# In[ ]:


corpus2


# In[ ]:


X_test = tfidf.transform(corpus2).toarray()


# In[ ]:


X_test.shape


# In[ ]:


X_test


# ## Predicting X_test Values

# In[ ]:


predicted_values = fake_news.predict(X_test)
predicted_values


# In[ ]:




