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
    
    train=[[]]*12
    tars=[[]]*12
    for  i,img in enumerate(imgs):
        #index=labels[i]
        img = np.reshape(img,[28,28])
        train[i%12]=train[i%12]+[[img]]     
        tars[i%12]= tars[i%12]+[labels[i]]

    test=[[]]*12
    ttars=[[]]*12

    for  i,img in enumerate(timgs):
        img = np.reshape(img,[28,28])
        test[i%12]=test[i%12]+[[img]]     
        ttars[i%12]= ttars[i%12]+[tlabels[i]]

    return [train,tars],[test,ttars]

class Net (nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 =nn.Conv2d(1,16,5)
        self.conv2 =nn.Conv2d(16,32,5)
        self.fc1 =nn.Linear(512,120)
        self.fc2 =nn.Linear(120,10)
        
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1)
    
def train():
    #net =Net()
    net=nnload()
    train,test=loaddata()
    net.cuda()
    for epoch in range(10):
        ii=0
        data=0
        for i,v in enumerate(train[0]):
            inp=v
            inp=th.Tensor(inp)
            inp = ag.Variable(inp)  
            inp=inp.cuda()
            out= net(inp)
            tar =th.Tensor(train[1][i])
            tar = ag.Variable(tar)
            tar=tar.long()
            tar=tar.cuda()
            criterion = nn.CrossEntropyLoss()
            criterion=criterion.cuda()
            #opt = op.SGD(net.parameters(),lr =0.01,momentum=0.9)
            opt = op.RMSprop(net.parameters(),lr =0.001,alpha=0.9)
            #opt = op.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.99))
            opt.zero_grad()
            loss=criterion(out,tar)
            loss=loss.cuda()
            loss.backward()
            opt.step()
            data+=loss.data
            ii+=1
        print ("loss ",data/ii)
    nnsave(net)
    
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
        for k,v in enumerate(predict):
            if test[1][i][k]==v:
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
    #train()
    