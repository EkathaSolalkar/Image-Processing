#!/usr/bin/env python
# coding: utf-8

# In[4]:


#restore damaged image
import numpy as np
import cv2

img = cv2.imread('cat_damaged.png')

mask = cv2.imread('cat_mask.png', 0)

dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

cv2.imwrite('cat_inpainted.png', dst)


# In[3]:


#removing object
from PIL import Image
import numpy as np	

img = Image.open('website.jfif').convert('RGB')
img_arr = np.array(img)
img_arr[0 : 400, 0 : 400] = (0, 0, 0)
img = Image.fromarray(img_arr)
img.show()


# In[6]:


#removing logo


# In[2]:


#restoration
import cv2
import numpy as np
from skimage import io     

frame = cv2.cvtColor(io.imread('crop.png'), cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(io.imread('nature.jpg'), cv2.COLOR_RGB2BGR)
mask = 255 * np.uint8(np.all(frame == [36, 28, 237], axis=2))
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt = min(contours, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(cnt)
frame[y:y+h, x:x+w] = cv2.resize(image, (w, h))
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


#removing text - not proper output
import numpy as np
import cv2 
img = cv2.imread('original_image.jpg')

print(img.shape)
h,w,c =  img.shape
txt = cv2.imread("cropped_image.jpg")
print(txt.shape)
hl,wl,cl  =  txt.shape

x1 = int(w/2-wl/2)
y1 = int(h/2-hl)
x2 = int(w/2+wl/2)
y2 =  int(h/2)
cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

cv2.imwrite("ReomveText.jpg",img)
cv2.imshow("textremove", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




