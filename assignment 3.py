#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
df = pd.read_csv('spam.csv')
df.head()


# In[4]:


df.groupby('Category').describe()


# In[5]:


df['spam'] = df['Category'].apply(lambda x:1 if x =='spam' else 0 )


# In[6]:


df.head()


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size = 0.25)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
V = CountVectorizer()
X_train_count = V.fit_transform(X_train.values)
X_train_count.toarray()[:3]


# In[9]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)


# In[10]:


emails = ['Hey Mohan, can we get together to watch football game tommorrow',
          'upto 20% discount on package, exclusive offer just for you. Don`t miss this reward!.']
email_count = V.transform(emails)
model.predict(email_count)


# In[11]:


X_test_count = V.transform(X_test)
model.score(X_test_count, y_test)


# In[12]:


from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('Vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train,y_train)
print(clf.score(X_test, y_test))
clf.predict(emails)


# In[ ]:




