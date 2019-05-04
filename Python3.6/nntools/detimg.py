# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:49:49 2019

@author: illya
"""

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import nntools as nn
from nntools import *

def sig(x):
    x=x*10
    y = 1.0 / (1.0 + np.exp(-x))
    y=np.uint8 (y*255)
    return y

def possesser(a):
    x,y=a.size
    a=a.convert('L')
    a=np.array(a)
    mid = np.mean(a)
    a=a-mid
    a=sig(a)
    a=Image.fromarray(a).convert('L')
    a=a.resize((28,28),Image.ANTIALIAS)
    return a

def det():
    lena=Image.open('4.png')
    lena=possesser(lena)
    lena=np.reshape(lena,[28,28])
    print (lena)
    plt.imshow(lena,cmap="gray") 
    plt.axis('on') 
    plt.show()
    lena=lena/np.std(lena)-1
    out1=nn.detect([lena])
    print ("nn =",out1)
    return (out1)

det()