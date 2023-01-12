#!/usr/bin/env python
# coding: utf-8

# In[5]:


#adding noise
import matplotlib.pyplot as plt
from skimage.util import random_noise
fruit_image=plt.imread('dog.png')
noisy_image=random_noise(fruit_image)
plt.title('Original image')
plt.imshow(fruit_image)
plt.show()


plt.title('Image after adding noise')
plt.imshow(noisy_image)
plt.show()


# In[6]:


#reducing noise
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
noisy_image=plt.imread('noise.jfif')
denoised_image=denoise_tv_chambolle(noisy_image,multichannel=True)
plt.title('Original image')
plt.imshow(noisy_image)
plt.show()


plt.title('Image after reducing noise')
plt.imshow(denoised_image)
plt.show()


# In[7]:


#reducing noise while preserving edges
import matplotlib.pyplot as plt
from skimage.restoration import denoise_bilateral
landscape_image=plt.imread('noise.jfif')
denoised_image=denoise_bilateral(landscape_image,multichannel=True)
plt.title('Original image')
plt.imshow(landscape_image)
plt.show()


plt.title('Image after reducing noise')
plt.imshow(denoised_image)
plt.show()


# In[ ]:




