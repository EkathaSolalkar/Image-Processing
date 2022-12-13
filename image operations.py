#!/usr/bin/env python
# coding: utf-8

# In[4]:


#converting from one format to another format 
from PIL import Image
img = Image.open('website.jpg')
img.save('website.jfif')


# In[8]:


#finding mode of an image
im=Image.open("img2.jfif")
print(im.mode)


# In[14]:


#image slicing
from PIL import Image
import numpy as np

im = np.array(Image.open('website.jpg').resize((256, 256)))
print(im.shape)

im_0 = im[:, :100]
print(im_0.shape)
Image.fromarray(im_0).save('slicing.jpg')


# In[15]:


#blending of images
from PIL import Image
img1=Image.open('cat.jfif')
img2=Image.open('tiger.jfif')

alphaBlended=Image.blend(img1,img2,alpha=.4)
alphaBlended.show()


# In[2]:


#cropping of an image
import cv2
img = cv2.imread("tiger.jfif")
y=0
x=0
h=200
w=380
crop_img = img[x:w, y:h]
cv2.imshow("Cropped", crop_img)
cv2.waitKey(0)


# In[4]:


#cropping of an image
#method2
from PIL import Image
im=Image.open('cat.jfif')
w,h=im.size
left=5
top=h/4
right=164
bottom=3*h/4

im1=im.crop((left,top,right,bottom))
im1.show()


# In[4]:


#negating of an image
import cv2
import numpy as np

img=cv2.imread('tiger.jfif')
print(img.dtype)
img_neg=255-img

cv2.imshow('negative',img_neg)
cv2.waitKey(0)


# In[2]:


#negating of an image
#method2
from PIL import Image
import matplotlib.pyplot as plt
img=Image.open('cat.jfif')

w,h=img.size
for i in range(w):
    for j in range(h):
        r,g,b=img.getpixel((i,j))
        r=255-r
        g=255-g
        b=255-b
        img.putpixel((i,j),(r,g,b))       
plt.axis('off')
plt.imshow(img)


# In[32]:


# drawing on an image
import numpy as np
import cv2

img =cv2.imread('cat.jfif')

cv2.line(img, (20, 160), (100, 160), (0, 0, 255), 10)
cv2.rectangle(img,(50,25), (200,300),(0,255,255),5)
cv2.circle(img, (20,50), 65, (255,0,0), -1)
cv2.imshow('dark', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[33]:


# draw text on an image
from PIL import Image,ImageDraw
img=Image.open('moon.jpg')
draw=ImageDraw.Draw(img)

txt="Image Processing"
draw.text((250,250),txt)
img.save('new.jfif')
img.show()


# In[23]:


#finding basic statistics of an image - mean
from PIL import Image, ImageStat

im = Image.open('moon.jpg')
stat = ImageStat.Stat(im)
print(stat.mean)


# In[31]:


#finding basic statistics of an image - median
from PIL import Image, ImageStat

im = Image.open('website.jpg')
stat = ImageStat.Stat(im)
print(stat.median)


# In[28]:


#finding basic statistics of an image - standard deviation
from PIL import Image, ImageStat

im = Image.open('moon.jpg')
stat = ImageStat.Stat(im)
print(stat.stddev)


# In[42]:


#histogram of an image
import cv2
import matplotlib.pyplot as plt
img=cv2.imread('cat.jfif')

plt.hist(img.ravel(),256,[0,256])
plt.show()


# In[ ]:




