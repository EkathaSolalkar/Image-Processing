#!/usr/bin/env python
# coding: utf-8

# In[1]:


#difference between two images
from PIL import Image,ImageChops
img1=Image.open("1img.jpg")
img2=Image.open("2img.jpg")
diff=ImageChops.difference(img1,img2)
diff.show()


# In[19]:


#image list only png files
import os
from os import listdir
folder_dir = "C:/Users/User/Desktop/images"
for images in os.listdir(folder_dir):
    if(images.endswith(".png")):
        print(images)


# In[20]:


#image list of all images
import os
from os import listdir
folder_dir = "C:/Users/User/Desktop/images"
for images in os.listdir(folder_dir):
    if(images.endswith("")):
        print(images)


# In[21]:


pip install glob


# In[2]:


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

images = []
for img_path in glob.glob('C:/Users/User/Desktop/images/*'):
    images.append(mpimg.imread(img_path))
    
plt.figure(figsize=(20,10))
columns=5
for i, image in enumerate(images):
    plt.subplot(len(images)/columns+1,columns,i+1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])


# In[ ]:





# In[ ]:




