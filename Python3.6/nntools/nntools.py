# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:34:05 2019

@author: illya
"""


import numpy as np
import time
import threading as th
import loaddata as ld
import numba as nb
import pickle as pk
import copy as cp 
#####多线程优化

def detect(inp):
    n=nnload()
    n.enter(inp)
    ret=n.forword()
    return np.argmax(ret)
    
    
def timetest(func,args,out):
    t1 = time.clock()
    ret=func(*args)
    t2 = time.clock()
    print (out,"[time]: %fs" % (t2 -t1))
    return ret

def nnsave(d):
    f = open('model.md.nn', 'wb')
    pk.dump(d, f)
    print("save success")
    
def nnload():
    f = open('model.md.nn', 'rb')
    d = pk.load(f)
    print("load success")
    return d

def train(n,args):
    n.train(*args)

class Task(th.Thread):
    
    def __init__(self,func,args,flag):
        th.Thread.__init__(self)
        self.func=func
        self.args=args
        self.flag=flag
        
    def run(self):
        self.ret=self.func(*self.args)
    
    def result(self):
        
        try:
            return [self.ret,self.flag]
        
        except Exception:
            return None
        
        


@nb.jit
def max_pool(dat,step):
    k=int(len(dat[0])/step)
    ret=np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            ii=step*i
            jj=step*j
            ret[i,j]=np.max(dat[ii:ii+step,jj:jj+step])
    
    return ret

@nb.jit
def multi_max_pool(dats,step):
    ret=[]
    for i in range (len(dats)):
        ret.append(max_pool(dats[i],step))   
    ret=np.asarray(ret)
    return ret

@nb.jit
def max_backpool(dat,step,y,loss):
    k=len(y)
    tret=np.zeros((k*step,k*step))
    for i in range(k):
        for j in range(k):
            ii=step*i
            jj=step*j
            tret[ii:ii+step,jj:jj+step]=np.where(dat[ii:ii+step,jj:jj+step]==y[i][j],loss[i][j],0)
    return tret


@nb.jit
def multi_max_backpool(dats,step,ys,loss):
    
    ret=[]
    for l in range(len(dats)):
        ret.append(mean_backpool(dats[l],step,ys[l],loss[l]))
    ret=np.asarray(ret)
    return ret

@nb.jit
def mean_pool(dat,step):
    k=int(len(dat[0])/step)
    ret=np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            ii=step*i
            jj=step*j
            ret[i,j]=np.mean(dat[ii:ii+step,jj:jj+step])
    
    return ret

@nb.jit
def multi_mean_pool(dats,step):
    ret=[]
    for i in range (len(dats)):
        ret.append(mean_pool(dats[i],step))   
    ret=np.asarray(ret)
    return ret 

@nb.jit
def mean_backpool(dat,step,y,loss):
    k=len(y)
    tret=np.zeros((k*step,k*step))
    for i in range(k):
        for j in range(k):
            ii=step*i
            jj=step*j
            #tret[ii:ii+step,jj:jj+step]=np.where(dat[ii:ii+step,jj:jj+step]==y[i][j],loss[i][j],0)
            tret[ii:ii+step,jj:jj+step]=loss[i][j]/step**2
    return tret


@nb.jit
def multi_mean_backpool(dats,step,ys,loss):
    
    ret=[]
    for l in range(len(dats)):
        ret.append(mean_backpool(dats[l],step,ys[l],loss[l]))
    ret=np.asarray(ret)
    return ret


    
@nb.jit 
def conv(dat,ker):

    ker_shape=ker.shape
    dat_shape=dat.shape
    k=dat_shape[1]-ker_shape[1]+1
    ret=np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            ret[i,j]=np.sum(np.multiply(dat[:,i:i+ker_shape[1],j:j+ker_shape[2]],ker))
     
    return ret

#####多任务卷积
@nb.jit
def multi_conv(dats,kers,bs):

    ret=[]  
    for l,ker in enumerate(kers): 
        ret.append(np.add(conv(dats,ker),bs[l]))
    ret=np.asarray(ret)
    return ret 

    
@nb.jit
def extend(arr,l):

    new_arr=np.zeros((len(arr)+l*2,len(arr)+l*2))
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            new_arr[i+l,j+l]=arr[i,j]
    return new_arr



def CrossEntropyLoss(y,tar):
    
    dy=-1/y
    y=-np.log(y)
    y=np.multiply(y,tar)
    dy=np.multiply(dy,tar)
    return [y,dy]



class Max_Pool:
    
    def __init__(self,step):
        
        self.step=step
        self.w=0
        self.b=0
        return
    
    def enter(self,i,):
        
        self.i =np.asarray(i)

    def forword(self):

        ret=multi_max_pool(self.i,self.step)
        self.y=ret

        return ret

    def backword(self,loss):
        self.dw=0
        self.db=0
        ret=multi_max_backpool(self.i,self.step,self.y,loss)        
        return ret
    
    def update(self,rate,bt1,bt2,lam):
        return
    
    def show(self):
        return 
    
    
class Mean_Pool:
    
    def __init__(self,step):
        
        self.step=step
        self.w=0
        self.b=0
        return
    
    def enter(self,i,):
        
        self.i =np.asarray(i)

    def forword(self):

        ret=multi_mean_pool(self.i,self.step)
        self.y=ret

        return ret

    def backword(self,loss):
        self.dw=0
        self.db=0
        ret=multi_mean_backpool(self.i,self.step,self.y,loss)        
        return ret
    
    def update(self,rate,bt1,bt2,lam):
        return
    
    def show(self):
        return 
    
    

class Shape:
    
    def __init__(self,shapes):
        self.ishape=shapes[0]
        self.oshape=shapes[1]
        self.w=0
        self.b=0
        return
    
    def enter(self,i,):
        self.i =np.asarray(i)
        self.ishape=self.i.shape

    def forword(self):
        self.y=0
        return np.reshape(self.i,self.oshape)

    def backword(self,loss):
        self.dw=0
        self.db=0
        return np.reshape(loss,self.ishape)
    
    def update(self,rate,bt1,bt2,lam):
        return
    
    def show(self):
        return 
    
    
    
class Relu:
    
    def enter(self,i):
        self.w=0
        self.b=0
        self.i = np.asarray(i)
        
    def forword(self):
        
        self.y = np.maximum(0,self.i)
        return self.y
    
    def backword(self,loss):
        self.dw=0
        self.db=0
        tem=np.where(self.y>=0,1,0)
        ret = np.multiply(tem,loss)
        return ret
    
    def update(self,rate,bt1,bt2,lam):
        return 
    
    def show(self):
        return 
    
    
    
class SoftMax:
    
    def enter(self,i):
        self.i =np.asarray(i)
        self.w=0
        self.b=0
    def forword(self):

        x=self.i
        x=x-np.mean(x)
        x=np.exp(x)
        self.y=x/(np.sum(x)+10**-8)
        return self.y
    
    def backword(self,loss):
        self.dw=0
        self.db=0
        y=np.asarray(self.y)
        ret = np.zeros((len(self.i),len(self.i)))
        for k in range(len(self.i)):
            for j in range(len(self.i)):
                if k==j:
                    ret[k,j]=y[k,0]*(1-y[j,0])
                else :
                    ret[k,j]=y[k,0]*(0-y[j,0])
        ret=np.dot(ret.T,loss)
        return ret
    
    def update(self,rate,bt1,bt2,lam):
        
        return
    
    def show(self):
        return 
    
    
"""
class Softmax:
    
    def enter(self,i):
        self.i =np.asarray(i)
        
    def forword(self):
        x=self.i
        x=x-np.mean(x)
        x=np.exp(x)
        self.y=x/np.sum(x)
        return self.y
    
    def backword(self,loss):
        dy=self.y-loss
        return dy
    
    def update(self,rate,bt1,bt2,lam):
        return
    
    def show(self):
        return 
"""    
    

class Sigmoid:
    
    def enter(self,i):
        self.i =np.asarray(i)
        self.w=0
        self.b=0
    def forword(self):
        x=self.i
        x=x-np.mean(x)
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y
    
    def backword(self,loss):
        dy=self.y*(1-self.y)
        dy=np.multiply(loss,dy)
        self.dw=0
        self.db=0
        return dy
    
    def update(self,rate,bt1,bt2,lam):
        return
    
    def show(self):
        return 
    
    
    
class Convolution:
    
    def __init__(self,shape):
        self.w=[]
        self.b=[]
        self.db=0
        self.dw=0
        for wn in range(shape[2]):
            ker=[]
            for x in range(shape[0]):
                w=[]
                
                for y in range (shape[1]):
                    l=np.random.randint(low=-1,high=2,size=shape[1])
                    w.append(l)
                    np.asarray(w)
                ker.append(np.asarray(w))
            self.b.append(np.random.randint(low=-1,high=2,size=1)[0])
            self.w.append(ker)
            
        self.w=np.asarray(self.w)   
        self.b=np.asarray(self.b)
        
        self.t=1
        self.wm=np.zeros(self.w.shape)
        self.wv=np.zeros(self.w.shape)
        self.bm=np.zeros(self.b.shape)
        self.bv=np.zeros(self.b.shape)
        
    def enter(self,i):
        
        self.i =np.asarray(i)
        
    def forword(self):
        
        ret=multi_conv(self.i,self.w,self.b)
        self.y=ret
        return ret

    def backword(self,loss):
        #print ("loss",loss)
        loss=np.asarray(loss)
        ker_shape=self.w.shape
        loss_shape=loss.shape
        
        tem=(loss_shape[0],loss_shape[-2]+(ker_shape[-2]-1)*2,loss_shape[-1]+(ker_shape[-1]-1)*2)
        tloss=np.zeros(tem)

        for i in range(loss_shape[0]):
            tloss[i]=extend(loss[i],ker_shape[-2]-1)
            
        tloss=np.asarray(tloss)

        kernal = self.w.copy()

        for i in range(ker_shape[0]):
            for j in range (ker_shape[1]):
                kernal[i,j]=np.rot90(kernal[i,j], 2)
            
        kernal=np.swapaxes(kernal,0,1)     

        ret=multi_conv(tloss,kernal,self.b)

        self.loss=ret
        
        dw=[]
        for kar in loss:
            tker=[]
            for i in self.i:
                tret=conv(np.asarray([i]),np.asarray([kar]))  
                tker.append(tret)
            dw.append(tker)
        self.dw=np.asarray(dw)
        self.db=np.asarray(np.sum(loss,axis=(-1,-2)))
        return ret
    
    def update(self,rate,bt1,bt2,lam):
        
        #正则化项
        self.w=(1-lam)*self.w
        self.b=(1-lam)*self.b
        
        #adam优化器
        self.wm=bt1*self.wm+(1-bt1)*self.dw
        self.wv=bt2*self.wv+(1-bt2)*self.dw**2
        dwm=self.wm/(1-bt1**self.t)
        dwv=self.wv/(1-bt2**self.t)
        dw=dwm/(dwv**0.5 + 1**-8)
        
        self.bm=bt1*self.bm+(1-bt1)*self.db
        self.bv=bt2*self.bv+(1-bt2)*self.db**2
        dbm=self.bm/(1-bt1**self.t)
        dbv=self.bv/(1-bt2**self.t)
        db=dbm/(dbv**0.5 + 1**-8)
        
        
        
        self.w=self.w - rate*dw
        self.b=self.b - rate*db
        
        self.t+=1

        return 0
    
    def show(self):
        print ("conv:\n",self.w)
    
    
    
class Layer:

    def __init__(self,shape):
        w=[]
        b=[]
        for x in range(shape[1]):
            l=np.random.randint(low=-1,high=2,size=shape[0])
            w.append(l)
            b.append(np.random.randint(low=-1,high=2,size=1)[0])
        self.w=np.asarray(w)
        self.b=np.asarray([b]).T
        
        #adam 优化器计数 初始动量和速度
        self.t=1
        self.wm=np.zeros(self.w.shape)
        self.wv=np.zeros(self.w.shape)
        self.bm=np.zeros(self.b.shape)
        self.bv=np.zeros(self.b.shape)
        
    def enter(self,i):
        self.i =np.asarray(i)

    def forword(self):
        ret = np.asarray(np.add(np.dot(self.w,self.i),self.b))
        self.y=ret
        return ret

    def backword(self,loss):
        self.db=loss
        l=np.dot(self.w.T,self.db)
        self.dw=np.dot(self.db,self.i.T)
        return l
    
    def update(self,rate,bt1,bt2,lam):
        
        
        #正则化项
        self.d=(1-lam)*self.w
        self.d=(1-lam)*self.b
        
        #adam 优化器
        self.wm=bt1*self.wm+(1-bt1)*self.dw
        self.wv=bt2*self.wv+(1-bt2)*self.dw**2
        dwm=self.wm/(1-bt1**self.t)
        dwv=self.wv/(1-bt2**self.t)
        dw=dwm/(dwv**0.5 + 1**-8)
        
        self.bm=bt1*self.bm+(1-bt1)*self.db
        self.bv=bt2*self.bv+(1-bt2)*self.db**2
        dbm=self.bm/(1-bt1**self.t)
        dbv=self.bv/(1-bt2**self.t)
        db=dbm/(dbv**0.5 + 1**-8)
        

        self.w =self.w -rate*dw
        self.b =self.b -rate*db
        
        self.t+=1
    
    
    def show(self):
        print ("layer:\n",self.w)
   

class Nn:

    def __init__(self,ls):
        self.ls=ls
  
    def enter(self,i):
        self.i=i
   
    def target(self,t):
        self.t=t
  
    def forword(self):
        tem_input =self.i
        for layer in self.ls:
            layer.enter(tem_input)
            tem_input=layer.forword()

        self.o=tem_input
        
        return self.o
    def backword(self,loss):
        for layer in self.ls[::-1]:
             
            loss=layer.backword(loss)
            
    def update(self,rate,bt1,bt2,lam):
        for layer in self.ls[::-1]:
            layer.update(rate,bt1,bt2,lam)
            
    
    def train(self,data,label,rate,bt1,bt2,lam):
        self.enter(data)
        self.target(label)
        self.forword()
        loss=CrossEntropyLoss(self.o,self.t)[1]
        self.backword(loss)
        #self.update(rate,bt1,bt2,lam)
            
    def verify(self,data,label):
        count=0
        num=0
        hit=0
        for i,v in enumerate(data):
            self.enter(v)
            self.forword()
            count=count+np.mean(CrossEntropyLoss(self.o,label[i])[0])
            if np.argmax(self.o)==np.argmax(label[i]):
                hit+=1
            num=num+1

        return count/num,hit/num
    
    def run(self,data,label,rate,test,tlabel,times,flag,bt1,bt2,lam,batch):
        ns=[]
        for i in range(batch):
            ns.append(cp.deepcopy(self))

        tasks=[]
        t1 = time.clock()
        for m in range(int(len(data)/batch)):          
            for i,v in enumerate(data[m*batch:(m+1)*batch]):
    
                tsk=Task(func=train,args=[ns[i],[v,label[m*batch+i],rate,bt1,bt2,lam]],flag=i)
                tsk.start()
                tasks.append(tsk)
                
            for tsk in tasks:
                tsk.join()
             
            tem=ns[0].add(ns[1:])
            tem.update(rate,bt1,bt2,lam)
            self.ls=tem.ls
            ns=[]
            tasks=[]
            for i in range(batch):
                ns.append(cp.deepcopy(tem))
                    
        loss,hit=n.verify(d,l)
        t2 = time.clock()
        print ("\r[train %02d]"%flag,"[time]: %9.8fs" % (t2 -t1),"[loss]: %11.10f"%loss,"[Hits rate]:%2.2f%%"%(hit*100),end=" "*10)
        return loss,hit
    def add(self,a):
        for i in range(len(self.ls)):
            for j in a:
                self.ls[i].dw=np.add(self.ls[i].dw,j.ls[i].dw)
                self.ls[i].db=np.add(self.ls[i].db,j.ls[i].db)
        
        for i in range(len(self.ls)):
            self.ls[i].dw=self.ls[i].dw/(len(a)+1)
            self.ls[i].db=self.ls[i].db/(len(a)+1)
        
        return cp.deepcopy(self)
    
    def show(self):
        for l in self.ls:
            l.show()


if __name__ == "__main__":

    file1= './mnist/train-images.idx3-ubyte'
    file2= './mnist/train-labels.idx1-ubyte'
    file3= './mnist/t10k-images.idx3-ubyte'
    file4= './mnist/t10k-labels.idx1-ubyte'
    imgs,data_head = ld.loadImageSet(file1)
    labels,labels_head = ld.loadLabelSet(file2)
    timgs,tdata_head = ld.loadImageSet(file3)
    tlabels,tlabels_head = ld.loadLabelSet(file4)
    imgs=imgs[0:50]
    labels=labels[0:50]
    d=[]
    l=[]
    
    for i in imgs:
        x=np.asarray([np.reshape(i,[28,28])])
        x=x/np.std(x)-1
        d.append(x)
    
    
    for i in labels:
        a=[0]*10
        a[i]=1
        l.append(np.asarray([a]).T)    
    d=np.asarray(d)
    l=np.asarray(l)
    
    l0=Convolution([1,5,16])
    l05=Relu()
    l18=Max_Pool(2)
    l2=Shape([[16,12,12],[16*12*12,1]])
    l3=Layer(shape=[16*12*12,60])
    l4=Sigmoid()
    l5=Layer(shape=[60,10])
    l6=SoftMax()
    n= Nn(ls = [l0,l05,l18,l2,l3,l4,l5,l6])

    print ("start")  
    for i in range(500):
        loss,hits=n.run(data=d,label=l,rate=0.05,test=d,tlabel=l,times=1,flag=i,bt1=0.9,bt2=0.999,lam=0.1,batch=10)
        if 1-hits<0.02:
            nnsave(n)
            break
    n.show()     
    for i in range(len(d)):
        n.enter(d[i])
        n.forword()
        print (np.argmax(l[i]),np.argmax(n.o))

"""
class Nn:

    def __init__(self,ls):
        self.ls=ls
  
    def enter(self,i):
        self.i=i
   
    def target(self,t):
        self.t=t
  
    def forword(self):
        tem_input =self.i
        for layer in self.ls:
            layer.enter(tem_input)
            tem_input=layer.forword()

        self.o=tem_input
        
        return self.o
    def backword(self,loss):
        for layer in self.ls[::-1]:                
            loss=layer.backword(loss)
            
    def update(self,rate,bt1,bt2,lam):
        for layer in self.ls[::-1]:
            layer.update(rate,bt1,bt2,lam)
            
    
    def train(self,data,label,rate,bt1,bt2,lam):
        
        for i,v in enumerate(data):
            self.enter(v)
            self.target(label[i])
            self.forword()
            #交叉熵损失函数需要搭配SoftMax运行
            loss=CrossEntropyLoss(self.o,self.t)[1]
            #loss=self.t
            self.backword(loss)
            self.update(rate,bt1,bt2,lam)
            
    def verify(self,data,label):
        count=0
        num=0
        hit=0
        for i,v in enumerate(data):
            self.enter(v)
            self.forword()
            count=count+np.mean(CrossEntropyLoss(self.o,label[i])[0])
            if np.argmax(self.o)==np.argmax(label[i]):
                hit+=1
            num=num+1
        return count/num,hit/num
    
    def run(self,data,label,rate,test,tlabel,times,flag,bt1,bt2,lam):
        t1 = time.clock()
        for i in range(times):
            self.train(data,label,rate,bt1,bt2,lam)
        loss,hit=self.verify(test,tlabel)
        t2 = time.clock()
        print ("\r[train %02d]"%flag,"[time]: %9.8fs" % (t2 -t1),"[loss]: %11.10f"%loss,"[Hits rate]:%2.2f%%"%(hit*100),end=" "*10)
        
        return loss,hit

    def show(self):
        for l in self.ls:
            l.show()


if __name__ == "__main__":

    file1= './mnist/train-images.idx3-ubyte'
    file2= './mnist/train-labels.idx1-ubyte'
    file3= './mnist/t10k-images.idx3-ubyte'
    file4= './mnist/t10k-labels.idx1-ubyte'
    imgs,data_head = ld.loadImageSet(file1)
    labels,labels_head = ld.loadLabelSet(file2)
    timgs,tdata_head = ld.loadImageSet(file3)
    tlabels,tlabels_head = ld.loadLabelSet(file4)
    imgs=imgs[1:500]
    labels=labels[1:500]
    d=[]
    l=[]
    
    for i in imgs:
        x=np.asarray([np.reshape(i,[28,28])])
        x=x/np.std(x)-1
        d.append(x)
    
    
    for i in labels:
        a=[0]*10
        a[i]=1
        l.append(np.asarray([a]).T)    
    
    
    l0=Convolution([1,5,16])
    l05=Relu()
    l08=Max_Pool(2)
    l2=Shape([[16,12,12],[16*12*12,1]])
    l3=Layer(shape=[16*12*12,120])
    l4=Sigmoid()
    l5=Layer(shape=[120,10])
    l6=SoftMax()
    n= Nn(ls = [l0,l08,l2,l3,l4,l5,l6])
    print ("start")
    
    for i in range(500):
        loss,hits=n.run(data=d,label=l,rate=0.01,test=d,tlabel=l,times=1,flag=i,bt1=0.9,bt2=0.999,lam=0.1)
        #nnsave(n)
        if 1-hits <0.01:
            break
        
    for i in range(len(d)):
        n.enter(d[i])
        n.forword()
        print (np.argmax(l[i]),np.argmax(n.o))
"""