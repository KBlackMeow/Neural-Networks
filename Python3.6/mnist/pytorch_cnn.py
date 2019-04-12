# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:16:40 2019

@author: illya
"""

import torch as th 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torch.optim as op
import loaddata as ld
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

def nnsave(d):
    f = open('model.md.cnn', 'wb')
    pk.dump(d, f)
    print("save success")
    
def nnload():
    f = open('model.md.cnn', 'rb')
    d = pk.load(f)
    print("load success")
    return d

def loaddata():
    file1= './mnist/train-images.idx3-ubyte'
    file2= './mnist/train-labels.idx1-ubyte'
    file3= './mnist/t10k-images.idx3-ubyte'
    file4= './mnist/t10k-labels.idx1-ubyte'
    imgs,data_head = ld.loadImageSet(file1)
    labels,labels_head = ld.loadLabelSet(file2)
    timgs,tdata_head = ld.loadImageSet(file3)
    tlabels,tlabels_head = ld.loadLabelSet(file4)
    
    train=[[]]*10
    for  i,img in enumerate(imgs):
        index=labels[i]
        img = np.reshape(img,[28,28])
        train[index]=train[index]+[[img]]     
    tars= labels
    
    test=[[]]*10
    for  i,img in enumerate(timgs):
        index=tlabels[i]
        img = np.reshape(img,[28,28])
        test[index]=test[index]+[[img]] 
    ttars= tlabels
    
    return [train,tars],[test,ttars]

class Net (nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 =nn.Conv2d(1,6,5)
        self.conv2 =nn.Conv2d(6,16,5)
        self.fc1 =nn.Linear(1*16*16,40)
        self.fc2 =nn.Linear(40,20)
        self.fc3 =nn.Linear(20,10)
        
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return x
    
def train():
    net =Net()
    train,test=loaddata()
    net.cuda()
    for epoch in range(50):
        ii=0
        data=0
        for i,v in enumerate(train[0]):
            inp=v
            inp=th.Tensor(inp)
            inp = ag.Variable(inp)  
            inp=inp.cuda()
            out= net(inp)
            tar =th.tensor([i]*len(v))
            tar = ag.Variable(tar)
            tar=tar.cuda()
            criterion = nn.CrossEntropyLoss()
            criterion=criterion.cuda()
            opt = op.SGD(net.parameters(),lr =0.001)
            opt.zero_grad()
            loss=criterion(out,tar)
            loss=loss.cuda()
            loss.backward()
            opt.step()
            data+=loss.data
            ii+=1
        print ("loss ",data/ii)
    nnsave()
    
def verify():
    train,test=loaddata()
    net=nnload()
    net.cuda()
    cot=0
    for i,v in enumerate(test[0]):
        plt.title(i)
        plt.imshow(np.reshape(v[421],[28,28]),cmap='gray')
        plt.show()
        inp=v
        inp=th.Tensor(inp)
        inp = ag.Variable(inp)  
        inp=inp.cuda()
        out= net(inp)
        _,predict = th.max(out.data,1)
        print (predict[421])
        for j in predict:
            if i==j:
                cot+=1
    print ("Hits %.2f"%(cot/100)+"%")


def detect(a):
    train,test=loaddata()
    net=nnload()
    net.cuda()
    inp=np.reshape(a,[1,1,28,28])
    inp=th.Tensor(inp)
    inp = ag.Variable(inp)  
    inp=inp.cuda()
    out= net(inp)
    _,predict = th.max(out.data,1)

    return predict[0]

if __name__=="__main__":
    verify()