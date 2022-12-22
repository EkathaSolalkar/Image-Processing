#!/usr/bin/env python
# coding: utf-8

# In[2]:


#arithmetic operations on image
import cv2
import matplotlib.pyplot as plt
img1=cv2.imread('cat.jfif')
img2=cv2.imread('tiger.jfif')

bitand=cv2.bitwise_and(img1,img2)
bitor=cv2.bitwise_or(img1,img2)
bitnot=cv2.bitwise_not(img1,img2)

cv2.imshow("Bitwise and",bitand)
cv2.imshow("Bitwise or",bitor)
cv2.imshow("Bitwise not",bitnot)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


#median filtering
import cv2
import numpy as np

img=cv2.imread('noise.jfif',0)
m,n=img.shape
new_img=np.zeros([m,n])
for i in range(1,m-1):
 for j in range(1,n-1):
    temp=[img[i-1,j-1],
         img[i-1,j],
         img[i-1,j+1],
         img[i,j-1],
         img[i,j],
         img[i,j+1],
         img[i+1,j-1],
         img[i+1,j],
        img[i+1,j+1]]
    temp=sorted(temp)
    new_img[i,j]=temp[4]       
new_img=new_img.astype(np.uint8)
cv2.imshow('Median filtered image',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


#average filtering
import cv2
import numpy as np
img=cv2.imread('Moon.jpg',0)
m,n=img.shape
mask=np.ones([3,3],dtype=int)
mask=mask/9

img_new=np.zeros([m,n])
for i in range(1,m-1):
    for j in range(1,n-1):
        temp=img[i-1,j-1]*mask[0,0]+img[i-1,j]*mask[0,1]+img[i-1,j+1]*mask[0,2]+img[i,j-1]*mask[1,0]+img[i,j]*mask[1,1]+img[i,j+1]*mask[1,2]+img[i+1,j-1]*mask[2,0]+img[i+1,j]*mask[2,1]+img[i+1,j+1]*mask[2,2]
        img_new[i,j]=temp
img_new=img_new.astype(np.uint8)
cv2.imshow('Blurred image',img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


from PIL import Image,ImageDraw,ImageFilter
im1=Image.open('bg image.png')
im2=Image.open('foreground.jfif')
mask_im=Image.new("L",im2.size,0)
draw=ImageDraw.Draw(mask_im)
draw.ellipse((20,50,400,300),fill=250)
mask_im_blur=mask_im.filter(ImageFilter.GaussianBlur(10))
back_im=im1.copy()
back_im.paste(im2,(0,0),mask_im_blur)
back_im.show()


# In[ ]:





# In[ ]:




