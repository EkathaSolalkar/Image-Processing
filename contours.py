#!/usr/bin/env python
# coding: utf-8

# In[10]:


#contours
import cv2
import numpy as np

image = cv2.imread('tiger.jfif')
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edged,
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


#montage 
import skimage.io
import skimage.util
a = skimage.io.imread('nature.jfif')
print(a.shape)
b = a // 2
c = a // 3
d = a // 4
m = skimage.util.montage([a, b, c, d], multichannel=True)
print(m.shape)
skimage.io.imsave('C:/Users/User/Desktop/mont.jpg', m)


# In[ ]:


#canvas 
import numpy as np
import matplotlib.pyplot as plt
import cv2
canvas=np.ones((600,600,3))
plt.imshow(canvas)

cv2.imread()

