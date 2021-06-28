#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


import tensorflow as tf


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


train = pd.read_csv(r'C:\Users\The ChainSmokers\Desktop\fake news\train.csv')


# In[5]:


train.head()


# In[6]:


train.shape


# In[7]:


train.isnull().sum()


# In[8]:


train.dropna(inplace=True)


# In[9]:


train.isnull().sum()


# In[10]:


message = train.copy()


# In[11]:


message.reset_index(inplace=True)


# In[12]:


message.head(15)


# In[13]:


message['title'][5]


# In[14]:


message.shape


# In[15]:


col1 = message['title']


# In[16]:


import re


# In[17]:


# importing nltk objects which will be in further use.
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps =  PorterStemmer()


# In[18]:


corpus=[]
for i in range(len(message)):
    review= re.sub('[^a-zA-Z]', ' ', col1[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)


# In[19]:


corpus


# In[20]:


corpus[5]


# In[21]:


len(corpus)


# In[ ]:





# # Word Embedding

# In[22]:


from tensorflow.keras.preprocessing.text import one_hot


# In[23]:


voc_size= 1000


# In[24]:


onehot_repr=[one_hot(words,voc_size)for words in corpus] 


# In[25]:


print(onehot_repr)


# In[26]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential


# In[27]:


sent_length= 15


# In[28]:


embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs[6])


# In[29]:


dim=50


# In[30]:


model=Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_length))
model.compile('adam','mse')


# In[31]:


X = model.predict(embedded_docs)


# In[32]:


X


# In[33]:


y = message['label'].values


# In[34]:


y


# # ANN Model

# In[35]:


from tensorflow.keras.layers import Dense


# In[36]:


# Adding a flattening layer
model.add(tf.keras.layers.Flatten())


# In[37]:


model.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[38]:


model.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[39]:


model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))


# In[40]:


# Compiling the ANN model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X,y, epochs=100)


# ## Test data

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


X_test = tfidf.transform(corpus2).toarray()


# In[ ]:


X_test.shape


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:





# In[ ]:





# In[ ]:




