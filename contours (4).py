#!/usr/bin/env python
# coding: utf-8

# In[1]:


#contours
import cv2
import numpy as np

image = cv2.imread('tiger.jfif')
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


#counting shapes
import matplotlib.pyplot as plt
def show_image_contour(image,contours):
        plt.figure
        for n,contour in enumerate(contours):
            plt.plot(contour[:,1],contour[:,0],linewidth=3)
        plt.imshow(image,interpolation='nearest',cmap='gray_r')
        plt.title('Contours')
        plt.axis('off')


# In[4]:


from skimage import measure,data
horse_image=data.horse()
contours=measure.find_contours(horse_image,level=0.8)
show_image_contour(horse_image,contours)


# In[17]:


#find contours of an image that is not binary
from skimage.io import imread
from skimage.filters import threshold_otsu
# from skimage.color import label2rgb
image_dices=imread('dice.png')

image_dices=color.rgb2gray(image_dices)
thresh=threshold_otsu(image_dices)

binary=image_dices > thresh
contours=measure.find_contours(binary,level=0.8)
show_image_contour(image_dices,contours)


# In[14]:


#count the dots in a dixe's image
import numpy as np

shape_contours=[cnt.shape[0]for cnt in contours]
max_dots_shape=50
dots_contours=[cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]
show_image_contour(binary,contours)
print('Dice`s dots number: {}.'.format(len(dots_contours)))


# In[ ]:




