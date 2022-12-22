#!/usr/bin/env python
# coding: utf-8

# In[41]:


#upsampling
import cv2
import matplotlib.pyplot as plt
img=cv2.imread('nature.jfif')
cv2.imshow("Size of image before pyrUp",img)

img=cv2.pyrUp(img)
cv2.imshow('Upsample',img)
plt.imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


#downsampling
import cv2
import matplotlib.pyplot as plt
img=cv2.imread('nature.jfif')
cv2.imshow("Size of image before pyrDown",img)

img=cv2.pyrDown(img)
cv2.imshow('Downsample',img)
plt.imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[29]:


#nearest neighbor interpolation
import cv2
import numpy as np
img = cv2.imread('img2.jfif')
near_img = cv2.resize(img,None, fx = 5, fy = 5, interpolation = cv2.INTER_NEAREST)
cv2.imshow('Near',near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[27]:


#bilinear neighbor interpolation
import cv2
import numpy as np
img = cv2.imread('img2.jfif')
bilinear_img = cv2.resize(img,None, fx = 5, fy = 5, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Bilinear',bilinear_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


#bilinear neighbor interpolation
import cv2
import numpy as np
img = cv2.imread('img2.jfif')
bicubic_img = cv2.resize(img,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Bicubic',bicubic_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[40]:


#quantization of image
from PIL import Image
import PIL
img=Image.open('images.png')
img=img.quantize(256)
img.show()


# In[ ]:




