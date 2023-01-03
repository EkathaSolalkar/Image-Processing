#!/usr/bin/env python
# coding: utf-8

# In[5]:


#difference between two images
from PIL import Image,ImageChops
img1=Image.open("1img.jpg")
img2=Image.open("2img.jpg")
diff=ImageChops.difference(img1,img2)
diff.show()


# In[ ]:




