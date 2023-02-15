# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 10:53:45 2018

@author: sean
"""
from math import sqrt
import math
import random as rd
from scipy.spatial import cKDTree as KDTree
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def euclidean(x, y):
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

def _fast_edges(G, radius, p):
    pos = nx.get_node_attributes(G, 'pos')
    nodes, coords = list(zip(*pos.items()))
    kdtree = KDTree(coords)  # Cannot provide generator.
    edge_indexes = kdtree.query_pairs(radius, p)
    edges = ((nodes[u], nodes[v]) for u, v in edge_indexes)
    return edges

# N is the number of nodes, and r is the radius
def initiate_graph(N,r,p):
    g=nx.Graph()
    g.add_nodes_from(range(N))
    nodes=g.nodes()
    pos = {v: [rd.random() for i in range(2)] for v in nodes}
    nx.set_node_attributes(g, pos, 'pos')
    for d in nx.nodes(g):
        if rd.random()<p:
            g.node[d]['strategy']=1
        else:
            g.node[d]['strategy']=0
    #strategy={v: rd.randint(0,1) for v in nodes}
    #nx.set_node_attributes(g,strategy,'strategy')
    edges=_fast_edges(g, r, 2)
    g.add_edges_from(edges)
    return g

# g is the graph, pm is the game, w is the selection intensity
def calculate_fitness(g,pm,w):
    payoff={}
    fitness={}
    for d1 in nx.nodes(g):
        payoff[d1]=0
        for d2 in nx.all_neighbors(g,d1):
            mid1=np.dot([g.node[d1]['strategy'],1-g.node[d1]['strategy']],pm)
            payoff[d1]+=np.dot(mid1,[g.node[d2]['strategy'],1-g.node[d2]['strategy']])
        fitness[d1]=1-w+w*payoff[d1]
    nx.set_node_attributes(g,payoff,'payoff')
    nx.set_node_attributes(g,fitness,'fitness')
    return g

# calculate the average fitness of each individual
# g is the graph, pm is the game, w is the selection intensity
def calculate_average_fitness(g,pm,w):
    payoff={}
    fitness={}
    for d1 in nx.nodes(g):
        payoff[d1]=0
        fitness[d1]=1
        for d2 in nx.all_neighbors(g,d1):
            mid1=np.dot([g.node[d1]['strategy'],1-g.node[d1]['strategy']],pm)
            payoff[d1]+=np.dot(mid1,[g.node[d2]['strategy'],1-g.node[d2]['strategy']])
        if g.degree(d1)!=0:
            fitness[d1]=1-w+w*payoff[d1]/float(g.degree(d1))
    nx.set_node_attributes(g,payoff,'payoff')
    nx.set_node_attributes(g,fitness,'fitness')
    return g


# selecte the individual for reproducation with a probability proportional to their fitness
def selected_birth(g):
    fitness=nx.get_node_attributes(g,'fitness')
    totalfit=sum(fitness.values())
    rand=rd.random()
    end=0
    for d in nx.nodes(g):
        end=end+g.node[d]['fitness']/float(totalfit)
        if rand<end:break
    return d


def link_number(g):
    cc=0
    cd=0
    dd=0
    for d1 in nx.nodes(g):
        for d2 in nx.all_neighbors(g,d1):
            if g.node[d1]['strategy']==g.node[d2]['strategy']:
                if g.node[d1]['strategy']==1:
                    cc+=1
                else:
                    dd+=1
            else:
                cd+=1
    cc1=cc/float(cc+cd+dd)
    cd1=cd/float(cc+cd+dd)
    dd1=dd/float(cc+cd+dd)
    return [cc1,cd1,dd1]


# The evolutionary process
# N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_game(N,r_1,p,r_2,c,w,u,M):
    g=initiate_graph(N,r_1,p)
    g=calculate_fitness(g,[[1,0],[1+c,c]],w)
    i=1
    result=[]
    while i<M:
        cooper=nx.get_node_attributes(g,'strategy')
        aver_cooper=sum(cooper.values())/float(g.order())
        result.append(aver_cooper)
        birth=selected_birth(g)
        k=i+N-1
        pos_k=[g.node[birth]['pos'][0]+r_2*(2*rd.random()-1),g.node[birth]['pos'][1]+r_2*(2*rd.random()-1)]
        g.add_node(k,pos=pos_k,strategy=g.node[birth]['strategy'],payoff=0,fitness=1)  
        for d in nx.nodes(g):
            if sqrt((g.node[d]['pos'][0] - pos_k[0]) ** 2+(g.node[d]['pos'][1] - pos_k[1]) ** 2)<r_1:
                g.add_edge(d,k)
        g.remove_edge(k,k)
        p=rd.random()
        if p<u/float(2):
            g.node[k]['strategy']=1-g.node[birth]['strategy']
        for d in nx.all_neighbors(g,k):
            mid1=np.dot([g.node[k]['strategy'],1-g.node[k]['strategy']],[[1,0],[1+c,c]])
            mid2=np.dot(mid1,[g.node[d]['strategy'],1-g.node[d]['strategy']])
            g.node[k]['payoff']+=mid2
            mid3=np.dot([g.node[d]['strategy'],1-g.node[d]['strategy']],[[1,0],[1+c,c]])
            mid4=np.dot(mid3,[g.node[k]['strategy'],1-g.node[k]['strategy']])
            g.node[d]['payoff']+=mid4
            g.node[d]['fitness']=1-w+w*g.node[d]['payoff']
        g.node[k]['fitness']=1-w+w*g.node[k]['payoff']
        nodes=g.nodes()
        death=rd.sample(nodes,1)[0]
        for d in nx.all_neighbors(g,death):
            mid1=np.dot([g.node[d]['strategy'],1-g.node[d]['strategy']],[[1,0],[1+c,c]])
            mid2=np.dot(mid1,[g.node[death]['strategy'],1-g.node[death]['strategy']])
            g.node[d]['payoff']-=mid4
            g.node[d]['fitness']=1-w+w*g.node[d]['payoff']
        g.remove_node(death)
        i+=1
    eout=[result,g]
    return eout


# The evolutionary process average degree
# N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_game_2(N,r_1,p,r_2,c,w,u,M):
    g=initiate_graph(N,r_1,p)
#    g_ini=g.copy()
    g=calculate_average_fitness(g,[[1,0],[1+c,c]],w)
    i=1
    result=[]
#    aver_deg=[]   
#    cc=[]
#    cd=[]
#    dd=[]
    while i<M:
#        if i==5000:
#            g_5000=g.copy()
#        if i==10000:
#            g_10000=g.copy()
#        if i==20000:
#            g_20000=g.copy()
#        if i==40000:
#            g_40000=g.copy()
#        if i==80000:
#            g_80000=g.copy()
#        if i==160000:
#            g_160000=g.copy()
#        if i==320000:
#            g_320000=g.copy()
#        if i==640000:
#            g_640000=g.copy()
        cooper=nx.get_node_attributes(g,'strategy')
        aver_cooper=sum(cooper.values())/float(g.order())
        result.append(aver_cooper)
#        aver_deg.append(nx.number_of_edges(g)/float(N))
#        cdc=link_number(g)
#        cc.append(cdc[0])
#        cd.append(cdc[1])
#        dd.append(cdc[2])
        birth=selected_birth(g)
        k=i+N-1
        pos_k=[g.node[birth]['pos'][0]+r_2*(2*rd.random()-1),g.node[birth]['pos'][1]+r_2*(2*rd.random()-1)]
        g.add_node(k,pos=pos_k,strategy=g.node[birth]['strategy'],payoff=0,fitness=1)  
        for d in nx.nodes(g):
            if sqrt((g.node[d]['pos'][0] - pos_k[0]) ** 2+(g.node[d]['pos'][1] - pos_k[1]) ** 2)<r_1:
                g.add_edge(d,k)
        g.remove_edge(k,k)
        p=rd.random()
        if p<u/float(2):
            g.node[k]['strategy']=1-g.node[birth]['strategy']
        for d in nx.all_neighbors(g,k):
            mid1=np.dot([g.node[k]['strategy'],1-g.node[k]['strategy']],[[1,0],[1+c,c]])
            mid2=np.dot(mid1,[g.node[d]['strategy'],1-g.node[d]['strategy']])
            g.node[k]['payoff']+=mid2
            mid3=np.dot([g.node[d]['strategy'],1-g.node[d]['strategy']],[[1,0],[1+c,c]])
            mid4=np.dot(mid3,[g.node[k]['strategy'],1-g.node[k]['strategy']])
            g.node[d]['payoff']+=mid4
            g.node[d]['fitness']=1-w+w*g.node[d]['payoff']/float(g.degree(d))
        if g.degree(k)!=0:
            g.node[k]['fitness']=1-w+w*g.node[k]['payoff']/float(g.degree(k))
        nodes=g.nodes()
        death=rd.sample(nodes,1)[0]
        for d in nx.all_neighbors(g,death):
            mid1=np.dot([g.node[d]['strategy'],1-g.node[d]['strategy']],[[1,0],[1+c,c]])
            mid2=np.dot(mid1,[g.node[death]['strategy'],1-g.node[death]['strategy']])
            g.node[d]['payoff']-=mid2
            if g.degree(d)!=0:
                g.node[d]['fitness']=1-w+w*g.node[d]['payoff']/float(g.degree(d))
        g.remove_node(death)
        i+=1
#    eout=[result,aver_deg]
    return result

# repeat the evolution process for K times
def rep_evol_game(N,r_1,p,r_2,c,w,u,M,K):
    result=[]
#    result2=[]
    for i in range(K):
        mid=evol_game_2(N,r_1,p,r_2,c,w,u,M)
#        mid1=mid[0]
        mid1=mid[4000:]
#       mid2=mid[1]
#        mid2=mid2[4000:]
        mid3=sum(mid1)/float(len(mid1))
#        mid4=sum(mid2)/float(len(mid2))
        result.append(mid3)
#        result2.append(mid4)
    return result
#    
# repeat the evolution process for various c
def rep_evol_gamec(N,r_1,p,r_2,c,u,M,K):
    result=[]
#    result2=[]
    w=[]
    for r in range(15):
        w=0.+0.01*r
        mid=rep_evol_game(N,r1,p,r_2,c,w,u,M,K)
#        mid1=mid[0]
#        mid2=mid[1]
        mid3=sum(mid)/float(len(mid))
#        mid4=sum(mid2)/float(len(mid2))
        result.append(mid3)
#        result2.append(mid4)
        r2.append(r1)
    return [r2,result]

# N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_game_compare1(N,r_1,p,c,w,u,M):
    g=initiate_graph(N,r_1,p)
#    g_ini=g.copy()
    g=calculate_average_fitness(g,[[1,0],[1+c,c]],w)
    i=1
    result=[]
#    result2=[]
    while i<M:
        cooper=nx.get_node_attributes(g,'strategy')
        aver_cooper=sum(cooper.values())/float(g.order())
        result.append(aver_cooper)
#        result2.append(nx.average_clustering(g))
        birth=selected_birth(g)
        k=i+N-1
        pos_k=[rd.random(),rd.random()]
        g.add_node(k,pos=pos_k,strategy=g.node[birth]['strategy'],payoff=0,fitness=1)  
        for d in nx.nodes(g):
            if sqrt((g.node[d]['pos'][0] - pos_k[0]) ** 2+(g.node[d]['pos'][1] - pos_k[1]) ** 2)<r_1:
                g.add_edge(d,k)
        g.remove_edge(k,k)
        p=rd.random()
        if p<u/float(2):
            g.node[k]['strategy']=1-g.node[birth]['strategy']
        for d in nx.all_neighbors(g,k):
            mid1=np.dot([g.node[k]['strategy'],1-g.node[k]['strategy']],[[1,0],[1+c,c]])
            mid2=np.dot(mid1,[g.node[d]['strategy'],1-g.node[d]['strategy']])
            g.node[k]['payoff']+=mid2
            mid3=np.dot([g.node[d]['strategy'],1-g.node[d]['strategy']],[[1,0],[1+c,c]])
            mid4=np.dot(mid3,[g.node[k]['strategy'],1-g.node[k]['strategy']])
            g.node[d]['payoff']+=mid4
            g.node[d]['fitness']=1-w+w*g.node[d]['payoff']/float(g.degree(d))
        if g.degree(k)!=0:
            g.node[k]['fitness']=1-w+w*g.node[k]['payoff']/float(g.degree(k))
        nodes=g.nodes()
        death=rd.sample(nodes,1)[0]
        for d in nx.all_neighbors(g,death):
            mid1=np.dot([g.node[d]['strategy'],1-g.node[d]['strategy']],[[1,0],[1+c,c]])
            mid2=np.dot(mid1,[g.node[death]['strategy'],1-g.node[death]['strategy']])
            g.node[d]['payoff']-=mid2
            if g.degree(d)!=0:
                g.node[d]['fitness']=1-w+w*g.node[d]['payoff']/float(g.degree(d))
        g.remove_node(death)
        i+=1
#    eout=[result,g_ini,g_1000,g_2000,g_4000,g]
    return result
#    
# N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_game_compare2(N,r_1,p,c,w,u,M):
    g=initiate_graph(N,r_1,p)
#    g_ini=g.copy()
    g=calculate_average_fitness(g,[[1,0],[1+c,c]],w)
    i=1
    result=[]
    result2=[]
    while i<M:
        cooper=nx.get_node_attributes(g,'strategy')
        aver_cooper=sum(cooper.values())/float(g.order())
        result.append(aver_cooper)
        result2.append(nx.average_clustering(g))
        birth=selected_birth(g)
        k=i+N-1
        g.add_node(k,pos=[0,0],strategy=g.node[birth]['strategy'],payoff=0,fitness=1) 
        nodes=g.nodes()
        death=rd.sample(nodes,1)[0]
        if death==k:
            g.remove_node(death)
        else:
            g.node[k]['pos']=[g.node[death]['pos'][0],g.node[death]['pos'][1]]
            for d in nx.all_neighbors(g,death):
                g.add_edge(d,k)
            p=rd.random()
            if p<u/float(2):
                g.node[k]['strategy']=1-g.node[birth]['strategy']
            for d in nx.all_neighbors(g,k):
                mid1=np.dot([g.node[k]['strategy'],1-g.node[k]['strategy']],[[1,0],[1+c,c]])
                mid2=np.dot(mid1,[g.node[d]['strategy'],1-g.node[d]['strategy']])
                g.node[k]['payoff']+=mid2
                mid3=np.dot([g.node[d]['strategy'],1-g.node[d]['strategy']],[[1,0],[1+c,c]])
                mid4=np.dot(mid3,[g.node[k]['strategy'],1-g.node[k]['strategy']])
                g.node[d]['payoff']+=mid4
                mid5=np.dot([g.node[d]['strategy'],1-g.node[d]['strategy']],[[1,0],[1+c,c]])
                mid6=np.dot(mid5,[g.node[death]['strategy'],1-g.node[death]['strategy']])
                g.node[d]['payoff']-=mid6
                g.node[d]['fitness']=1-w+w*g.node[d]['payoff']/float(g.degree(d))
            if g.degree(k)!=0:
                g.node[k]['fitness']=1-w+w*g.node[k]['payoff']/float(g.degree(k))
            g.remove_node(death)
        i+=1
#    eout=[result,g_ini,g_1000,g_2000,g_4000,g]
    return [result,result2]


## N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_network(N,r_1,p,M):
    g=initiate_graph(N,r_1,p)
    g_ini=g.copy()
    i=1
    result2=[]
    while i<M:
        result2.append(nx.average_clustering(g))
        nodes=g.nodes()
        birth=rd.sample(nodes,1)[0]
        k=i+N-1
        pos_k=[rd.random(),rd.random()]
        g.add_node(k,pos=pos_k)  
        for d in nx.nodes(g):
            if sqrt((g.node[d]['pos'][0] - pos_k[0]) ** 2+(g.node[d]['pos'][1] - pos_k[1]) ** 2)<r_1:
                g.add_edge(d,k)
        g.remove_edge(k,k)
        nodes=g.nodes()
        death=rd.sample(nodes,1)[0]
        g.remove_node(death)
        i+=1
#    eout=[result,g_ini,g_1000,g_2000,g_4000,g]
    return [result2,g_ini,g]

# repeat the evolution process for K times
def rep_evol_game2(N,r_1,p,c,w,u,M,K):
    result=[]
#    result2=[]
    for i in range(K):
        mid=evol_game_compare1(N,r_1,p,c,w,u,M)
#        mid1=mid[0]
 #       mid1=mid1[10000:]
#        mid2=mid[1]
#        mid2=mid2[10000:]
        mid3=sum(mid)/float(len(mid))
 #       mid4=sum(mid2)/float(len(mid2))
        result.append(mid3)
#        result2.append(mid4)
    return result


# repeat the evolution process for various c
def rep_evol_gamec2(N,r_1,p,w,u,M,K):
    result=[]
#    result2=[]
    c2=[]
    for c in range(10):
        c1=0.1*c
        mid=rep_evol_game2(N,r_1,p,c1,w,u,M,K)
#        mid1=mid[0]
#        mid2=mid[1]
        mid3=sum(mid)/float(len(mid))
#        mid4=sum(mid2)/float(len(mid2))
        result.append(mid3)
 #       result2.append(mid4)
        c2.append(c1)
    return [c2,result]



def rep_geo_graph(N,r_1,M):
    result=[]
    i=1
    while i<M:
        g=nx.random_geometric_graph(N,r_1)
        result.append(nx.number_of_edges(g)/float(N))
        i+=1
    return result


def draw_degree(g):
    result=[]
    degree=nx.degree_histogram(g)
    x=range(len(degree))
    y=[z/float(sum(degree)) for z in degree]
    for i in x:
        result.append(sum(y[i:]))
    plt.loglog(x,result)
    return result

#rslt=rep_geo_graph(100,0.12,1000)
    
rslt=rep_evol_game(200,0.5,0.5,0.1,0.2,0.2,0.01,1000000,10)
#g1=rslt[0]
#result=[]
#degree=nx.degree_histogram(g1)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=0$")
#g2=rslt[1]
#result=[]
#degree=nx.degree_histogram(g2)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=0.5\times 10^4$")
#g3=rslt[2]
#result=[]
#degree=nx.degree_histogram(g3)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=1\times 10^4$")
#g4=rslt[3]
#result=[]
#degree=nx.degree_histogram(g4)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=2\times 10^4$")
#g5=rslt[4]
#result=[]
#degree=nx.degree_histogram(g5)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=4\times 10^4$")
#g6=rslt[5]
#result=[]
#degree=nx.degree_histogram(g6)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=8\times 10^4$")
#g7=rslt[6]
#result=[]
#degree=nx.degree_histogram(g7)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=16\times 10^4$")
#g8=rslt[7]
#result=[]
#degree=nx.degree_histogram(g8)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=32\times 10^4$")
#g9=rslt[8]
#result=[]
#degree=nx.degree_histogram(g9)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=64\times 10^4$")
#g10=rslt[9]
#result=[]
#degree=nx.degree_histogram(g10)
#x=range(len(degree))
#y=[z/float(sum(degree)) for z in degree]
#for i in x:
#    result.append(sum(y[i:]))
#plt.loglog(x,result,label="$t=1\times 10^6$")

#draw_degree(g1)
#draw_degree(g2)
#draw_degree(g3)
#draw_degree(g4)
#draw_degree(g5)
#draw_degree(g6)
#draw_degree(g7)
#draw_degree(g8)
#draw_degree(g9)
#draw_degree(g10)
#plt.figure(1)
#plt.plot(rslt[0],'m-',label="Cooperation Frequency")
#plt.plot(rslt[7],'r-',label="proportion of CC links")
#plt.plot(rslt[8],'b-',label="proportion of CD links")
#plt.plot(rslt[9],'g-',label="proportion of DD links")
#plt.xlabel('Generations',fontsize=14,fontweight="heavy",fontname='arial')
#
#plt.figure(2)
#g1=rslt[1]
#nodelist1=[]
#nodelist2=[]
#for d in nx.nodes(g1):
#    if g1.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g1,pos=nx.get_node_attributes(g1,'pos'),nodelist=nodelist1,node_color='m',node_size=20)
#nx.draw_networkx_nodes(g1,pos=nx.get_node_attributes(g1,'pos'),nodelist=nodelist2,node_color='c',node_size=20)
#nx.draw_networkx_edges(g1,pos=nx.get_node_attributes(g1,'pos'))
#
#plt.figure(3)
#g2=rslt[2]
#nodelist1=[]
#nodelist2=[]
#for d in nx.nodes(g2):
#    if g2.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g2,pos=nx.get_node_attributes(g2,'pos'),nodelist=nodelist1,node_color='m',node_size=20)
#nx.draw_networkx_nodes(g2,pos=nx.get_node_attributes(g2,'pos'),nodelist=nodelist2,node_color='c',node_size=20)
#nx.draw_networkx_edges(g2,pos=nx.get_node_attributes(g2,'pos'))
#
#plt.figure(4)
#g3=rslt[3]
#nodelist1=[]
#nodelist2=[]
#for d in nx.nodes(g3):
#    if g3.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g3,pos=nx.get_node_attributes(g3,'pos'),nodelist=nodelist1,node_color='m',node_size=20)
#nx.draw_networkx_nodes(g3,pos=nx.get_node_attributes(g3,'pos'),nodelist=nodelist2,node_color='c',node_size=20)
#nx.draw_networkx_edges(g3,pos=nx.get_node_attributes(g3,'pos'))
#
#plt.figure(5)
#g4=rslt[4]
#nodelist1=[]
#nodelist2=[]
#for d in nx.nodes(g4):
#    if g4.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g4,pos=nx.get_node_attributes(g4,'pos'),nodelist=nodelist1,node_color='m',node_size=20)
#nx.draw_networkx_nodes(g4,pos=nx.get_node_attributes(g4,'pos'),nodelist=nodelist2,node_color='c',node_size=20)
#nx.draw_networkx_edges(g4,pos=nx.get_node_attributes(g4,'pos'))
#
#plt.figure(6)
#g5=rslt[5]
#nodelist1=[]
#nodelist2=[]
#for d in nx.nodes(g5):
#    if g5.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g5,pos=nx.get_node_attributes(g5,'pos'),nodelist=nodelist1,node_color='m',node_size=20)
#nx.draw_networkx_nodes(g5,pos=nx.get_node_attributes(g5,'pos'),nodelist=nodelist2,node_color='c',node_size=20)
#nx.draw_networkx_edges(g5,pos=nx.get_node_attributes(g5,'pos'))
#
#plt.figure(7)
#g6=rslt[6]
#nodelist1=[]
#nodelist2=[]
#for d in nx.nodes(g6):
#    if g6.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g6,pos=nx.get_node_attributes(g6,'pos'),nodelist=nodelist1,node_color='m',node_size=20)
#nx.draw_networkx_nodes(g6,pos=nx.get_node_attributes(g6,'pos'),nodelist=nodelist2,node_color='c',node_size=20)
#nx.draw_networkx_edges(g6,pos=nx.get_node_attributes(g6,'pos'))


#rslt2=evol_game_compare2(100,0.12,0.5,0.2,0.2,0.01,10000)

#g1=rslt[1]
#rslt=rep_evol_game2(100,0.12,0.5,0,0.2,0.01,30000,80)
#rslt=rep_evol_game(100,0.12,0.5,0.12,0.2,0,0.01,30000,100)
#rslt=rep_evol_gamec2(100,0.12,0.5,0.2,0.01,20000,20)
##rslt=evol_game_2(100,0.12,0.5,0.12,0.2,0.2,0.01,15000)
#plt.figure(1)
#plt.plot(rslt[0],rslt[1],'ro-')
#plt.figure(2)
#plt.plot(rslt[0],rslt[2],'ro-')
#plt.xlabel('Generation',fontsize=14,fontweight='heavy')
#plt.ylabel('Fraction of Cooperation',fontsize=14,fontweight='heavy')
#g1=rslt1[2]
#nodelist1=[]
#nodelist2=[]
#for d in nx.nodes(g1):
#    if g1.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g1,pos=nx.get_node_attributes(g1,'pos'),nodelist=nodelist1,node_size=20)
#nx.draw_networkx_nodes(g1,pos=nx.get_node_attributes(g1,'pos'),nodelist=nodelist2,node_color='b',node_size=20)
#nx.draw_networkx_edges(g1,pos=nx.get_node_attributes(g1,'pos'))
#
#plt.figure(2)
#g2=rslt1[3]
#nodelist1=[]
#nodelist2=[]
#plt.figure(3)
#for d in nx.nodes(g2):
#    if g2.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g2,pos=nx.get_node_attributes(g2,'pos'),nodelist=nodelist1,node_size=20)
#nx.draw_networkx_nodes(g2,pos=nx.get_node_attributes(g2,'pos'),nodelist=nodelist2,node_color='b',node_size=20)
#nx.draw_networkx_edges(g2,pos=nx.get_node_attributes(g2,'pos'))
#
#plt.figure(3)
#g3=rslt2[2]
#nodelist1=[]
#nodelist2=[]
#plt.figure(4)
#for d in nx.nodes(g3):
#    if g3.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g3,pos=nx.get_node_attributes(g3,'pos'),nodelist=nodelist1,node_size=20)
#nx.draw_networkx_nodes(g3,pos=nx.get_node_attributes(g3,'pos'),nodelist=nodelist2,node_color='b',node_size=20)
#nx.draw_networkx_edges(g3,pos=nx.get_node_attributes(g3,'pos'))
#
#plt.figure(4)
#g4=rslt2[3]
#nodelist1=[]
#nodelist2=[]
#plt.figure(5)
#for d in nx.nodes(g4):
#    if g4.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g4,pos=nx.get_node_attributes(g4,'pos'),nodelist=nodelist1,node_size=20)
#nx.draw_networkx_nodes(g4,pos=nx.get_node_attributes(g4,'pos'),nodelist=nodelist2,node_color='b',node_size=20)
#nx.draw_networkx_edges(g4,pos=nx.get_node_attributes(g4,'pos'))
#
#g5=rslt[5]
#nodelist1=[]
#nodelist2=[]
#plt.figure(6)
#for d in nx.nodes(g5):
#    if g5.node[d]['strategy']==1:
#        nodelist1.append(d)
#    else:
#        nodelist2.append(d)
#nx.draw_networkx_nodes(g5,pos=nx.get_node_attributes(g5,'pos'),nodelist=nodelist1,node_size=20)
#nx.draw_networkx_nodes(g5,pos=nx.get_node_attributes(g5,'pos'),nodelist=nodelist2,node_color='b',node_size=20)
#nx.draw_networkx_edges(g5,pos=nx.get_node_attributes(g5,'pos'))
