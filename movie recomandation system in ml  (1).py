#!/usr/bin/env python
# coding: utf-8

# # import require labraries and dataset 

# In[1]:


import pandas as pd 
import numpy as np
import itertools
from  mlxtend.preprocessing import TransactionEncoder
import seaborn as sns


# In[2]:


dt=pd.read_csv("Amazon - Movies and TV Ratings.csv")


# In[3]:


dt.head()


# In[4]:


dt.describe()


# In[5]:


dt.shape


# # avrage rating of all movies

# In[6]:


dt.describe().T['mean'].sum()/len(dt.columns)-1


# In[7]:


dt.describe().mean()


# # movie with highest rating

# In[8]:


dt.drop('user_id',axis=1).sum().sort_values(ascending=False)[:1].to_frame() 


# # five top rated movies

# In[9]:


dt.describe().T['mean'].sort_values(ascending=False)[0:5]


# # top 5 movies with the least audience

# In[10]:


dt.describe().T['count'].sort_values(ascending=True)[0:5]


# In[11]:


dt.fillna(value=0,inplace=True)


# In[12]:


#dt=dt.astype(bool).astype(int)


# In[13]:


#dt.drop("user_id",axis=1,inplace=True)


# In[14]:


dt


# In[15]:


#any movie with such null value 

an=dt.isnull().sum()/dt.shape[0]
an[an==1]


# In[16]:


dt.sum()


# In[18]:


from surprise import Reader
from surprise import accuracy
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise.model_selection import cross_validate


# In[19]:


df_melt = dt.melt(id_vars = dt.columns[0],value_vars=dt.columns[1:],var_name="Movies",value_name="Rating")


# In[20]:


df_melt


# In[21]:


rd = Reader()
data = Dataset.load_from_df(df_melt.fillna(0),reader=rd)
data


# In[22]:


trainset, testset = train_test_split(data,test_size=0.25)


# In[23]:


#Using SVD (Singular Value Descomposition)
svd = SVD()
svd.fit(trainset)


# In[24]:


pred = svd.test(testset)


# In[25]:


accuracy.rmse(pred)


# In[26]:


accuracy.mae(pred)


# In[28]:


cross_validate(svd, data, measures = ['RMSE', 'MAE'], cv = 3, verbose = True)


# In[29]:


def repeat(ml_type,dframe):
    rd = Reader()
    data = Dataset.load_from_df(dframe,reader=rd)
    print(cross_validate(ml_type, data, measures = ['RMSE', 'MAE'], cv = 3, verbose = True))
    print("--"*15)
    usr_id = 'A3R5OBKS7OM2IR'
    mv = 'Movie1'
    r_u = 5.0
    print(ml_type.predict(usr_id,mv,r_ui = r_u,verbose=True))
    print("--"*15)


# In[30]:


repeat(SVD(),df_melt.fillna(df_melt['Rating'].mean()))
#repeat(SVD(),df_melt.fillna(df_melt['Rating'].median()))

************************end******************************