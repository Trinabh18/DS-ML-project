#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd


# In[3]:


os.chdir("D:\DATA SCIENCE PROJECTS DATASET")


# In[14]:


messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages


# In[6]:


#data cleaing
import re
import nltk
nltk.download('stopwords')


# In[7]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[ ]:


ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[27]:




# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# In[20]:


X


# In[11]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[12]:


# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)



# In[13]:


y_pred


# In[ ]:




