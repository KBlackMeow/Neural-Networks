# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:49:49 2019

@author: illya
"""


import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
#import numpy as np
import svdnn
#import cnn
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

lena = mpimg.imread('7.png') 
#print (lena[:,:,2])
print(lena.shape) 
#lena=lena.max(axis=-1)
#lena=lena.mean(axis=2)
#gravity= np.array([0.299,0.587,0.114])
#lena=np.dot(lena,gravity)
lena=rgb2gray(lena)
#print (lena)
plt.imshow(lena,cmap="gray") 
plt.axis('on') 
plt.show()
out=svdnn.detect(lena.reshape(28*28))
print (out[-1])
