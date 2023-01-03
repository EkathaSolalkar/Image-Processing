#!/usr/bin/env python
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import imageio
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
pic=imageio.imread('tiger.jfif')
plt.figure(figsize=(6,6))
plt.title('image enhamncement')
plt.imshow(pic)
plt.axis('off');


# In[4]:


negative=255-pic
plt.figure(figsize=(6,6))
plt.imshow(negative)
plt.axis('off');


# In[7]:


import numpy as np

pic=imageio.imread('tiger.jfif')
gray=lambda rgb : np.dot(rgb[...,:3],[0.2999,0.587,0.114])
gray=gray(pic)

max_=np.max(gray)

def log_transform():
    return(255/np.log(1+max_))*np.log(1+gray)
plt.figure(figsize=(5,5))
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))
plt.axis('off');


# In[8]:


pic=imageio.imread('tiger.jfif')
gamma=2.2

gamma_correction=((pic/255)**(1/gamma))
plt.figure(figsize=(5,5))
plt.imshow(gamma_correction)
plt.axis('off');


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import imageio
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
pic=imageio.imread('C:/Users/User/Desktop/images')

#negative
negative=255-pic

#grayscale
import numpy as np

gray=lambda rgb : np.dot(rgb[...,:3],[0.2999,0.587,0.114])
gray=gray(pic)

max_=np.max(gray)

def log_transform():
    return(255/np.log(1+max_))*np.log(1+gray)

#gamma transform
gamma=2.2
gamma_correction=((pic/255)**(1/gamma))

fig,ax=plt.subplots(4,4,figsize=(5,5), sharey=True)
ax[0].imshow(pic)
ax[0].set_title('origin')
ax[1].imshow(negative)
ax[1].set_title('negative')
ax[2].imshow(log_transform(),cmap=plt.get_cmap(name='gray'))
ax[2].set_title('log')
ax[3].imshow(gamma_correction)
ax[3].set_title('gamma')

plt.axis('off');


# In[ ]:




