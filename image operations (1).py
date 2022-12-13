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


# In[66]:


# image slicing
#method2
from skimage.io import imshow,imread
import matplotlib.pyplot as plt
img1=imread('website.jpg')
imshow(img1)

fig,ax=plt.subplots(1,3,figsize=(6,4), sharey=True)
ax[0].imshow(img1[:, 0:150])
ax[0].set_title('First slice')

ax[1].imshow(img1[:, 150:300])
ax[1].set_title('second slice')

ax[2].imshow(img1[:, 300:500])
ax[2].set_title('Third slice');


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


# In[58]:


# draw text on an image
from PIL import Image,ImageDraw,ImageFont

img = Image.open('moon.jpg')
I1 = ImageDraw.Draw(img)

fnt=ImageFont.truetype('arial.ttf', 50)
I1.text((28, 36), "Image processing", font=fnt, fill=(255, 0, 0))

img.show()
img.save("moon1.png")


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

red_hist=cv2.calcHist([img],[0],None,[256],[0,255])
green_hist=cv2.calcHist([img],[1],None,[256],[0,255])
blue_hist=cv2.calcHist([img],[2],None,[256],[0,255])

plt.hist(img.ravel(),256,[0,256])
plt.show()


# In[ ]:




