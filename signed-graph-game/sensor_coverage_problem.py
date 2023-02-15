import numpy as np
import math
import random as rd


#############################################
#definition of the density function W(s)
def density_function(u):
    dens={}
    for i in range(6):
        for j in range(4):
            if j==0:
                w=26
            elif j==2:
                w=25
            elif j==1 or j==3:
                w=24
            s1=[0.15+0.3*i,0.15+0.3*j]
            rslt=math.exp(-w/9*(math.pow((s1[0]-u[0]),2)+math.pow((s1[1]-u[1]),2)))
            dens[i,j]=rslt
    return dens

# determine the number of collaborative players for each square
# here the input is the strategy profiles of all players
def col_number(a):
    n={}
    for i in range(6):
        for j in range(4):
            n[i,j]=0
    for d in a:
        n[d]=n[d]+1
        if d[0]+1<6:
            n[d[0]+1,d[1]]+=1
        if d[0]-1>-1:
            n[d[0]-1,d[1]]+=1
        if d[1]+1<4:
            n[d[0],d[1]+1]+=1
        if d[1]-1>-1:
            n[d[0],d[1]-1]+=1
    return n

#determine the utility profile of one player given the strategies of other players
def get_utility(a,w):
    u={}
    for i in range(6):
        for j in range(4):
            u[i,j]=0
            b=a[:]
            b.append((i,j))
            n=col_number(b)
            u[i,j]+=w[i,j]/n[i,j]
            if i+1<6:
                u[i,j]+=w[i+1,j]/n[i+1,j]
            if i-1>-1:
                u[i,j]+=w[i-1,j]/n[i-1,j]
            if j+1<4:
                u[i,j]+=w[i,j+1]/n[i,j+1]
            if j-1>-1:
                u[i,j]+=w[i,j-1]/n[i,j-1]
    return u

#add dynamic attribute-the probability
def add_dynamic_attributes():
    p1={}
    p2={}
    p3={}
    for i in range(6):
        for j in range(4):
            p1[i,j]=1/24
            p2[i,j]=1/24
            p3[i,j]=1/24
    p=[p1,p2,p3]
    return p
    
#one step updating of strategies and probabilities
def one_step_updating(a,w,p):
    ind=rd.randint(0,2)
    del a[ind]
    u=get_utility(a,w)
    p1=p[ind]
    aver=0
    for i in range(6):
        for j in range(4):
            aver+=p1[i,j]*u[i,j]
    for i in range(6):
        for j in range(4):
            p1[i,j]=p1[i,j]*u[i,j]/aver
    p[ind]=p1
    x=rd.uniform(0,1)
    cum_prob=0
    for i,j in zip(range(6),range(4)):
        cum_prob+=p1[i,j]
        if x<cum_prob:break
    a.insert(ind,(i,j))
    return (a,p)

# compute the potetial given the strategy profile
def potential(a,w):
    n=col_number(a)
    total=0
    for i in range(6):
        for j in range(4):
            if n[i,j]>0:
                for k in range(n[i,j]):
                    g=k+1
                    total+=w[i,j]/g
    return total

# The general updating process for T steps
def general_updating(a,u,T):
    w=density_function(u)
    p=add_dynamic_attributes()
    seq=[]
    t=potential(a,w)
    seq.append(t)
    for i in range(T):
        mid=one_step_updating(a,w,p)
        a=mid[0]
        p=mid[1]
        t=potential(a,w)
        seq.append(t)
    return (seq,a,p)
############################################


