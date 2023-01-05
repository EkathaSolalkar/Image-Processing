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


#montage - method 1
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


# In[4]:


#montage - method 2
import cv2
from PIL import Image
from skimage import io


IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400

def create_collage(images):
    images = [io.imread(img) for img in images]
    images = [cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)) for image in images]
    if len(images) > 2:
        half = len(images) // 2
        h1 = cv2.hconcat(images[:half])
        h2 = cv2.hconcat(images[half:])
        concat_images = cv2.vconcat([h1, h2])
    else:
        concat_images = cv2.hconcat(images)
    image = Image.fromarray(concat_images)

    # Image path
    image_name = "montage.png"
    image = image.convert("RGB")
    image.save(f"{image_name}")
    return image_name
images=["1img.jpg","Moon.jpg","unnamed.jpg","nature.jpg"]
#image1 on top left, image2 on top right, image3 on bottom left,image4 on bottom right
create_collage(images)


# In[5]:


#collage (montage)
import cv2
import numpy as np

image1=cv2.imread("1img.jpg")
image2=cv2.imread("Moon.jpg")
image3=cv2.imread("unnamed.jpg")
image4=cv2.imread("nature.jpg")

image1=cv2.resize(image1,(200,200))
image2=cv2.resize(image2,(200,200))
image3=cv2.resize(image3,(200,200))
image4=cv2.resize(image4,(200,200))

Horizontal1=np.hstack([image1,image2])
Horizontal2=np.hstack([image3,image4])
Vertical_attachment=np.vstack([Horizontal1,Horizontal2])
cv2.imshow("Final Collage",Vertical_attachment)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#canvas 
import numpy as np
import matplotlib.pyplot as plt
import cv2
canvas=np.ones((600,600,3))
plt.imshow(canvas)

cv2.imread()


# In[6]:


#removing watermark
import cv2
import numpy as np

img = cv2.imread("Watermarks.png")
alpha = 2.0
beta = -160
new = alpha * img + beta
new = np.clip(new, 0, 255).astype(np.uint8)
cv2.imshow("Removed image", new)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




