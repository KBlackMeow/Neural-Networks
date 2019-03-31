# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:34:05 2019

@author: illya
"""


import numpy as np
import time

"""
sigmoid 函数
sigmoid1是sigmod的导数
"""
def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y

def sigmoid1(y):
    y=y*(1-y)
    return y


class Neuron:
    def __init__(self,w,b):
        self.w=np.array(w)
        self.b=b

    def output(self):
        return self.w.dot(self.i.T) + self.b
       
    def set_in(self,i):
        self.i=i
        
    def set_hi(self,h):
        self.h=h
                
    def grad_des(self,r,g):
        self.w=self.w - r*g*self.i.T
        self.b=self.b - r*g
       
class Neu_layer:
    def __init__(self,i,n):
        ns=[]
        for x in range(n):
            l=np.random.randn(i)
            l=l/np.linalg.norm(l,ord=1)
            ns.append(Neuron(l,np.random.randn(1)[0]))
        self.ns=ns
        
    def set_in(self,i):
        self.i =i
        
    def set_hi(self,h):
        self.h=h 
        
    def output(self):
        temout=[]
        for neu in self.ns:
            neu.set_in(self.i)
            tem=neu.output()
            neu.set_hi(sigmoid(tem))
            temout.append(neu.h)
        temout= np.array(temout)
        self.set_hi(temout)
        return temout
    
    def gd(self,rate,g):
        tg=np.zeros(len(self.ns[0].w))
        g=g*sigmoid1(self.h)
        for i in range (len(self.ns)):
            tg=tg+self.ns[i].w*g[i]
            self.ns[i].grad_des(rate,list(g)[i])
        return tg
class Neu_net:
    def __init__(self,defin):
        ls=[]
        for i in range(len(defin)-1):
            ls.append(Neu_layer(defin[i],defin[i+1]))
        self.ls=ls

    def set_in(self,i):
        self.i=i   
    def set_out(self,output_val):
        self.o=output_val
       
    def output(self):
        tem_input =self.i
        for layer in self.ls:
            layer.set_in(tem_input)
            tem_input=layer.output()
        return tem_input
    
    def o_grad(self):
        y=self.output()
        g=(y-self.o)
        return g
    
    def train(self,rate):   
        grad=self.o_grad()
        for layer in self.ls[::-1]:
            grad=layer.gd(rate,grad)

class Neu_layer1:

    def __init__(self,i,n):
        w=[]
        b=[]
        for x in range(n):
            l=np.random.randn(i)
            l=l/np.linalg.norm(l,ord=1)
            w.append(l)
            b.append(np.random.randn(1)[0])
        self.w=np.mat(w)
        self.b=np.mat(b).T

    def set_in(self,i):
        self.i =np.mat(i)

    def set_hi(self,h):
        self.h=h

    def output(self):
        temout =sigmoid(np.array(np.dot(self.w,self.i)+ self.b))
        self.set_hi(temout)
        return temout

    def gd(self,rate,g):
        g=rate*np.multiply(g,sigmoid1(np.array(self.h)))
        tg=np.dot(self.w.T,g)
        dw=np.dot(g,self.i.T)
        self.w=self.w -dw
        self.b =self.b -g
        return tg
    
class Neu_net1:

    def __init__(self,defin):
        ls=[]
        for i in range(len(defin)-1):
            ls.append(Neu_layer1(defin[i],defin[i+1]))
        self.ls=ls
  
    def set_in(self,i):
        self.i=np.mat(i).T
   
    def set_out(self,output_val):
        self.o=output_val
  
    def output(self):
        tem_input =self.i
        for layer in self.ls:
            layer.set_in(tem_input)
            tem_input=layer.output()
        return tem_input

    def o_grad(self):
        y=self.output()
        g=(y-self.o)
        return g

    def train(self,rate):   
        grad=self.o_grad()
        for layer in self.ls[::-1]:
            grad=layer.gd(rate,grad)
            

        
        
"""
四种运算
00表示非运算
01表示与运算
10表示或运算啊
11表示异或运算
输入 xx       xx      x
    运算符    输入数据  真实的输出值

"""
data=[
      [[0,1,0,0,0],[0,1,0,1,0],[0,1,1,0,0],[0,1,1,1,1]],
      [[1,0,0,0,0],[1,0,0,1,1],[1,0,1,0,1],[1,0,1,1,1]],
      [[0,0,0,0,1],[0,0,0,1,0]],
      [[1,1,0,0,0],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]]
]
#d= Neu_net([4,3,1])
d= Neu_net1([4,3,1])

def train():
    for i in range (1000):
        for dats in data:
            for dat in dats :
                d.set_in(np.array(dat[0:4]))
                d.set_out(dat[-1])
                d.train(1)

def test():
    count=0
    num=0
    hit=0
    for dats in data:
        for dat in dats :
            d.set_in(np.array(dat[0:4]))
            out=d.output()
            if np.linalg.norm(dat[-1]-out[0]) < 0.5:
                hit=hit+1
            count=count+(out-dat[-1])**2
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
    i=0
    for dats in data:
        for dat in dats :
            d.set_in(np.array(dat[0:4]))
            out=d.output()
            print("[test %02d] [input]:"%i,dat,"[output]: %.8f [value]: %d"%(out,out>0.5)) 
            i+=1
           
for i in range(10):
    var=run(i)
show()
