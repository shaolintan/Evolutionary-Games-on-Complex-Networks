import numpy as np
import math


def initial(N,e):
    x=np.random.rand(N)
    y=np.random.rand(N)
    for i in range(N):
        Nei[i]=[]
        for j in range(N):
            if math.sqrt((x[i]-x[j])^2+(y[i]-y[j])^2)<e:
                Nei[i].append(j)
    s=np.random.randint(2,size=N)
    return (x,y,Nei,s)

def payoff(N,Nei,s,g,w):
    for i in range(N):
        p[i]=0
        for j in Nei[i]:
            p[i]=p[i]+[s[i],1-s[i]]*g*[s[j],1-s[j]]'
        p[i]=1-w+w*p[i]
    return p

def select_birth(p):
    
        


