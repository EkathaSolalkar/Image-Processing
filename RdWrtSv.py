#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
image=cv2.imread('website.jpg',0)
cv2.imshow('Display Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


cv2.imwrite('C:/Users/User/Desktop/image1.jpg',image)


# In[5]:


from PIL import Image
filepath="website.jpg"
img=Image.open(filepath)
width=img.width
height=img.height
print('The height of the image is :',height)
print('The width of the image is :',width)


# In[ ]:


import numpy

img = cv2.imread("website.jpg")
print('No of Channel is: ' + str(img.ndim))

cv2.imshow("Channel", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


from PIL import Image
filepath="website.jpg"
img=Image.open(filepath)
new=img.resize((75,75))
new


# In[9]:


from PIL import Image
filepath="website.jpg"
img=Image.open(filepath)
width=30
height=28
new=img.resize((width,height),Image.ANTIALIAS)
new


# In[4]:


import matplotlib.image as image
img=image.imread('website.jpg')
print('The Shape of the image is:',img.shape)
print('The image as array is:')
print(img)


# In[6]:


import cv2
img=cv2.imread('website.jpg',2)
ret,n_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
n=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("BINARY",n_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


img=cv2.imread('website.jpg')
B,G,R=cv2.split(img)
print(B)
print(G)
print(R)


# In[11]:


cv2.imshow("Blue",B)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:


cv2.imshow("Green",G)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


cv2.imshow("Red",R)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[20]:


import cv2
import matplotlib.image as image
from PIL import Image
filepath="website.jpg"
img=Image.open(filepath)
width=img.width
height=img.height
print('The height of the image is :',height)
print('The width of the image is :',width)
width=30
height=28
new=img.resize((width,height),Image.ANTIALIAS)
new


# In[ ]:




