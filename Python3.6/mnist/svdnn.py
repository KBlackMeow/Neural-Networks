# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:55:48 2019

@author: illya
"""
import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt 
import loaddata as ld
import time
import pickle as pk

def svd(a,x,y):
    a=np.reshape(a,[x,y])
    v,l,q = np.linalg.svd(np.mat(a))
    ret=np.zeros(shape=(x,y))
    for i in range(5):
        ret=ret+np.dot(np.dot(v[:,i],l[i]),q[i,:])
    ret=np.array(ret)
    ret=np.reshape(ret,x*y)
    return ret
    
def activeR(x):
    if x<0:
        x=0        
    return x

def activeR1(y):
    if y > 0:
        y=1
    return y

def activeM(x):
    list_a = np.array(x.T)[0]
    M=np.max(list_a)
    exps = np.exp(x-M)
    return exps/np.sum(exps)

def activeM1(x,y):
    y=np.array(y.T)
    list_a = np.array(x.T)[0]
    max_index = np.argwhere(np.max(list_a)==list_a)
    list_a[max_index[0][0]]=list_a[max_index[0][0]]-y[0][max_index[0][0]]
    ret = np.mat(list_a).T
    return ret

def randlist(i,n):
    w=[]
    b=[]
    for x in range(n):
            l=np.random.randn(i)
            l=l/np.linalg.norm(l,ord=1)
            w.append(l)
            b.append(np.random.randn(1)[0])
    return w,b

def Convolution(dat,dsize,ker,ksize):
    ret=[]
    dat=np.array(dat)
    ker=np.array(ker)
    x=(dsize[0]-ksize[0]+1)
    y=(dsize[1]-ksize[1]+1)
    for i in range(x*y):
        cot=0;
        r=int(i/x)
        s=i%y
        for j in range(ksize[0]):
            for k in range(ksize[1]):
                cot=cot+dat[r*dsize[1]+s+j*dsize[1]+k]*ker[j*ksize[1]+k]
        cot=activeR(cot)
        ret.append(cot)
    return ret

def Pool(dat,dsize,psize):
    ret=[]
    x=int(dsize[0]/psize[0])
    y=int(dsize[1]/psize[1])
    for i in range(x*y):
        cot=0;
        r=int(i/x)
        s=i%y
        for j in range(psize[0]):
            for k in range(psize[1]):                
                if cot<dat[r*dsize[1]*psize[0]+s*psize[1]+j*dsize[1]+k]:
                    cot=dat[r*dsize[1]*psize[0]+s*psize[1]+j*dsize[1]+k]
        ret.append(cot)
    return ret
            
    
def encode(a):
    ret=[0]*10
    ret[a]=1
    return np.array(ret)

def decode(a):
    count=0
    for i,v in enumerate(a):
        count=count+i*v
    return count
   
class NN:

    def __init__(self,df):
        ls=[]
        for i in range(len(df)-1):
            w,b=randlist(df[i],df[i+1])
            w=np.mat(w)
            b=np.mat(b).T
            ls.append([w,b])
        self.ls=ls
        
    def show(self):
        for i in self.ls:
            print("w=",i[0],"\nb=",i[1])
  
    def get_out(self,inp):
        hi=[]
        hi.append(inp)
        for i,l in enumerate(self.ls):
            tem=np.add(np.dot(l[0],hi[i]),l[1])
            tem=activeM(tem)
            hi.append(tem)
        return hi

    def train(self,inp,y,r):
        hi=self.get_out(inp)

        for i in range(len(self.ls)):
            l=self.ls[-(i+1)]
            g=activeM1(np.array(hi[-(i+1)]),y)
            tg=np.dot(l[0].T,g)
            dw=np.dot(g,hi[-(i+2)].T)
            l[0]=l[0] - np.dot(dw,r)
            l[1]=l[1] - np.dot(g,r)
            g=tg
            
def nnsave(d):
    f = open('model.md.svd', 'wb')
    pk.dump(d, f)
    print("save success")
    
def nnload():
    f = open('model.md.svd', 'rb')
    d = pk.load(f)
    print("load success")
    return d

Con_ker=np.array([
        [1,0,0,0,1,0,0,0,1],
        #[1,-1,0,1,-1,0,1,-1,0],
        #[0,0,-1,0,1,0,-1,0,0],
        #[0,1,-1,0,-1,0,0,1,0]
        
])





"""
plt.figure("Image") # 图像窗口名称
plt.axis('on') # 关掉坐标轴为 off
"""

def get_data(a,b,igs,lbs):
    x=[]
    y=[]
    for i in range(a,b+1):
        img =igs[i,:]
        lab =lbs[i]
        img=svd(np.array(img),28,28)
        x.append(np.array(img))
        y.append(encode(lab))     

    return x,y

def train(x,y,d):
    for i in range(len(x)):
        x[i] = x[i] /np.linalg.norm(x[i])
        c=x[i]
        d.train(np.mat(c).T,np.mat(y[i]).T,0.1)

def test(x,y,d):
        count=0
        num=0
        hit=0
        for i in range(len(x)):
            x[i] = x[i] /np.linalg.norm(x[i])
            c=x[i]
            out=d.get_out(np.mat(c).T)
            v=np.abs(decode(y[i])-np.argwhere(out[-1]==np.max(out[-1]))[0][0])
            if v < 0.1:
                hit=hit+1
            count=count+v
            num=num+1
        return count/num,hit/num
    
def run(times,d,x,y,a,b):
        
        t1 = time.clock()
        train(x,y,d)
        var,hits=test(a,b,d)
        t2 = time.clock()
        print ("[train %02d]"%times,"[time]: %fs" % (t2 -t1),"[var]: %.8f"%var,"[hits]: %2.2f"%(hits*100))
        return var

def start(imgs,labels,timgs,tlabels):
    d=NN([784,10])  
    #d=nnload()
    x,y=get_data(1,59999,imgs,labels)
    a,b=get_data(1,9999,timgs,tlabels)    
    for i in range(20):
        run(i,d,x,y,a,b)
    nnsave(d)


def verify(timgs,tlabels):
    d=nnload()
    index=0
    dx,dy=get_data(index,index+9999,timgs,tlabels)   
    j=0
    for i in range(len(dx)):
        dx[i] = dx[i] /np.linalg.norm(dx[i])
        out=d.get_out(np.mat(dx[i]).T)
        if (np.argwhere(out[-1]==np.max(out[-1]))[0][0]==tlabels[index+i]):
            j+=1
    print ("Hist = %.2f"%(j/100))
        
def detect(a):
    d=nnload()
    dat=np.array(a)
    ret=svd(dat,28,28)
    plt.title("svd")
    plt.imshow(np.reshape(ret,[28,28]),cmap='gray')
    plt.show()
    ret=ret/np.average(ret)
    inp=np.mat(ret).T
    out=d.get_out(inp)
    return out

if __name__=="__main__" :
    file1= './mnist/train-images.idx3-ubyte'
    file2= './mnist/train-labels.idx1-ubyte'
    file3= './mnist/t10k-images.idx3-ubyte'
    file4= './mnist/t10k-labels.idx1-ubyte'
    imgs,data_head = ld.loadImageSet(file1)
    labels,labels_head = ld.loadLabelSet(file2)
    timgs,tdata_head = ld.loadImageSet(file3)
    tlabels,tlabels_head = ld.loadLabelSet(file4)
    #start(imgs,labels,timgs,tlabels)
    verify(timgs,tlabels)