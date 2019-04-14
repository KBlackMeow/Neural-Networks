# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:49:49 2019

@author: illya
"""

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import pytorch_cnn as cnn
def possesser(a):
    #a=a.tobytes()
    a=np.reshape(a,(28*28))
    for i,v in enumerate(a):
        if v<128:
            a[i]=0x00
        else :
            a[i]=0xff
    return np.reshape(a,(28,28))
lena=Image.open('0.png')
lena=lena.resize((28,28),Image.ANTIALIAS)
lena=lena.convert('L')
lena = np.array(lena)
lena=possesser(lena)
#lena = np.array(lena)
print (lena)
plt.imshow(lena,cmap="gray") 
plt.axis('on') 
plt.show()
out=cnn.detect(lena)
print (out)
