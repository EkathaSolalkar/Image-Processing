#!/usr/bin/env python
# coding: utf-8

# In[4]:


from PIL import Image
import os 
os.getcwd()


# In[5]:


os.listdir()


# In[6]:


#separate the name and extension of the files
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        print(fn, "&", fext)


# In[7]:


#Looping over the image files
for f in os.listdir("."):
    if f.endswith(".jpg"):
        print(f)


# In[8]:


# Creating new Directory using OS library
os.mkdir('NewExtnsn')


# In[9]:


for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        i.save("NewExtnsn/{}.pdf".format(fn))


# In[10]:


# Creating new multiple Directories using OS library
os.makedirs('resize//small')
os.makedirs('resize//tiny')


# In[11]:


size_small = (600,600) # small images of 600 X 600 pixels
size_tiny = (200,200)  # tiny images of 200 X 200 pixels
for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        i.thumbnail(size_small)
        i.save("resize/small/{}_small{}".format(fn, fext))
        i.thumbnail(size_tiny)
        i.save("resize/tiny/{}_tiny{}".format(fn, fext))


# In[12]:


# Creating new Directory using OS library
os.mkdir('rotate')


# In[13]:


for f in os.listdir("."):
    if f.endswith(".jpg"):
        i = Image.open(f)
        fn, fext = os.path.splitext(f)
        im = i.rotate(90)
        im.save("rotate/{}_rot.{}".format(fn, fext))


# In[21]:


#Displaying new created directories and files within 
fold_dir='C:/Users/HP/Image Processing/'
fold_dir


# In[22]:


os.listdir(fold_dir)


# In[24]:


fold_dir1='C:/Users/HP/Image Processing/resize'
os.listdir(fold_dir1)


# In[28]:


small_fold='C:/Users/HP/Image Processing/resize/small'
os.listdir(small_fold)


# In[29]:


tiny_fold='C:/Users/HP/Image Processing/resize/tiny'
os.listdir(tiny_fold)


# In[25]:


fold_dir2='C:/Users/HP/Image Processing/rotate'
os.listdir(fold_dir2)


# In[26]:


fold_dir3='C:/Users/HP/Image Processing/NewExtnsn'
os.listdir(fold_dir3)


# In[ ]:




