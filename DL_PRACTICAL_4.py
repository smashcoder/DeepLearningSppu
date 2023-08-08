#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install theano')
import theano
from theano import *
import theano.tensor as T
import numpy as np
import pandas as pd
from theano import function


# In[2]:


v1 = T.dscalar()
v2 = T.scalar()


# In[3]:


sres = v1-v2


# In[4]:


ares = v1+v2


# In[5]:


calcsres = theano.function([v1,v2],sres)
calcares = theano.function([v1,v2],ares)
calcares(12,23)
calcsres(13,12)
x = T.dmatrix('x')
y = T.dmatrix('y')


# In[6]:


z = x+y
func = function([x,y],z)
m1 = [
[1,2],
[3,4]
]
m2 = [
[4,5],
[6,7]
]
func(m1,m2)


# In[ ]:




