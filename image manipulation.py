#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image,ImageFilter
import matplotlib.pyplot as plt

my_image=Image.open('cat.jfif')
sharp=my_image.filter(ImageFilter.SHARPEN)
sharp.save('C:/Users/User/Desktop/image_sharpen.jpg')
sharp.show()
plt.imshow(sharp)
plt.show()


# In[3]:


img=Image.open('cat.jfif')
plt.imshow(img)
plt.show()

flip=img.transpose(Image.FLIP_LEFT_RIGHT)
flip.save('C:/Users/User/Desktop/image_flip.jpg')
plt.imshow(flip)
plt.show()


# In[19]:


im=Image.open('cat.jfif')
width,height=im.size

im1=im.crop((50,25,200,175))
im1.show()
plt.imshow(im1)
plt.show()


# In[ ]:




