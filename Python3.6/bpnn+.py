# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:28:45 2019

@author: illya
"""
import numpy as np
import time
def randlist(i,n):
    w=[]
    b=[]
    for x in range(n):
            l=np.random.randn(i)
            l=l/np.linalg.norm(l,ord=1)
            w.append(l)
            b.append(np.random.randn(1)[0])
    return w,b

def active(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y

def active1(y):
    y=y*(1-y)
    return y

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
            tem=active(tem)
            hi.append(tem)
        return hi

    def g(self):
        y=self.out()
        g=(y-self.o)
        return g

    def train(self,inp,y,r):
        hi=self.get_out(inp)
        lost=hi[-1]-y
        g=lost
        for i in range(len(self.ls)):
            l=self.ls[-(i+1)]
            a=active1(np.array(hi[-(i+1)]))
            g=np.dot(r,np.multiply(g,a))
            tg=np.dot(l[0].T,g)
            dw=np.dot(g,hi[-(i+2)].T)
            l[0]=l[0] -dw
            l[1]=l[1] -g
            g=tg
            
if __name__ =="__main__":
    data=[
          [[0,1,0,0,0],[0,1,0,1,0],[0,1,1,0,0],[0,1,1,1,1]],
          [[1,0,0,0,0],[1,0,0,1,1],[1,0,1,0,1],[1,0,1,1,1]],
          [[0,0,0,0,1],[0,0,0,1,0]],
          [[1,1,0,0,0],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]]
    ]
    d= NN([4,3,1])
    
    def formatd(d):
        retx=[]
        rety=[]
        for dats in data:
            for dat in dats :
                retx.append(np.mat(dat[0:4]).T)
                rety.append(np.mat(dat[-1]).T)
        return retx,rety
    
    x,y=formatd(data)
    
    def train():
        for j in range(1000):
            for i in range(len(x)):
                d.train(x[i],y[i],1)
    
    def test():
        count=0
        num=0
        hit=0
        for i in range(len(x)):
            d.train(x[i],y[i],1)
            out=d.get_out(x[i])
            v=np.linalg.norm(y[i]-out[-1])
            if v < 0.5:
                hit=hit+1
            count=count+v
            num=num+1
        return count/num,hit/num
    
    def run(times):
        t1 = time.clock()
        train()
        var,hits=test()
        t2 = time.clock()
        print ("[train %02d]"%times,"[time]: %fs" % (t2 -t1),"[var]: %.8f"%var,"[hits]: %2.2f"%(hits*100))
        return var
    
    def show():
        j=0
        for i in range(len(x)):
            d.train(x[i],y[i],1)
            out=d.get_out(x[i])
            print("[test %02d] [input]:"%j,x[i].T,y[i].T,"[output]: ",out[-1]) 
            j+=1
               
    for i in range(10):
        var=run(i)
    show()
