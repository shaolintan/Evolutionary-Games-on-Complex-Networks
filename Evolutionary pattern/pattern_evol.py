import numpy as np
import math
import random as rd

# random growth of the population and its boundary
def random_grow(popul,boundary,d):
    direct=np.random.randint(0,4)   
    theta=[0,math.pi/float(2),math.pi,-math.pi/float(2)]
    dim=len(boundary[direct])
    loc=np.random.randint(0,dim)
    add=[0,0]
    add[0]=boundary[direct][loc][0]+d*math.cos(theta[direct])
    add[1]=boundary[direct][loc][1]+d*math.sin(theta[direct])
    boundary[direct][loc]=add
    popul.append(add)
    if direct==0 or direct==2:
        comp=[]
        for bd in boundary[1]:
            if bd[0]==add[0]:
                comp.append(bd)
                break
        if len(comp)==0:
            boundary[1].append(add)
        elif comp[0][1]<add[1]:
            boundary[1].append(add)
            boundary[1].remove(comp[0])
        comp=[]
        for bd in boundary[3]:
            if bd[0]==add[0]:
                comp.append(bd)
                break
        if len(comp)==0:
            boundary[3].append(add)
        elif comp[0][1]>add[1]:
            boundary[3].append(add)
            boundary[3].remove(comp[0])
    if direct==1 or direct==3:
        comp=[]
        for bd in boundary[0]:
            if bd[1]==add[1]:
                comp.append(bd)
                break
        if len(comp)==0:
            boundary[0].append(add)
        elif comp[0][0]<add[0]:
            boundary[0].append(add)
            boundary[0].remove(comp[0])
        comp=[]
        for bd in boundary[2]:
            if bd[1]==add[1]:
                comp.append(bd)
                break
        if len(comp)==0:
            boundary[2].append(add)
        elif comp[0][0]>add[0]:
            boundary[2].append(add)
            boundary[2].remove(comp[0])
    return (popul,boundary)

# pattern evolutionary process
def evol_pattern(popul,boundary,d,N):
    for i in range(N):
        result=random_grow(popul,boundary,d)
        popul=result[0]
        boundary=result[1]
    return (popul,boundary)

#


