#!/usr/bin/env python
# coding: utf-8

# In[1]:


#inverting image code
from PIL import Image
from PIL import ImageFilter
import os

def main():
    inPath ="C:/Users/User/Desktop/images"
    outPath ="C:/Users/User/Desktop/IP"
    for imagePath in os.listdir(inPath):
        inputPath = os.path.join(inPath, imagePath)
        img = Image.open(inputPath)
        fullOutPath = os.path.join(outPath, 'invert_'+imagePath)
        img.rotate(90).save(fullOutPath)
        print(fullOutPath)

if __name__ == '__main__':
    main()


# In[ ]:




