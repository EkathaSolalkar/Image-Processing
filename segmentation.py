#!/usr/bin/env python
# coding: utf-8

# In[3]:


#superpixel segmentation
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
face_image=plt.imread('face.jfif')
segments=slic(face_image,n_segments=400)
segmented_image=label2rgb(segments,face_image,kind='avg')
plt.title('Original image')
plt.imshow(face_image)
plt.show()


plt.title('segmenteed image, 400 superpixels')
plt.imshow(segmented_image)
plt.show()


# In[ ]:




