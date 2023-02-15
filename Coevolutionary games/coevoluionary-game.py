# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:58:50 2019

@author: sean
"""

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

# The evolutionary process average degree
# N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_game_2(N,r_1,p,r_2,c,w,u,M):
    g=initiate_graph(N,r_1,p)
    g=calculate_average_fitness(g,[[1,0],[1+c,c]],w)
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
    return result

# repeat the evolution process for K times
def rep_evol_game(N,r_1,p,r_2,c,w,u,M,K):
    result=[]
    for i in range(K):
        mid=evol_game_2(N,r_1,p,r_2,c,w,u,M)
        mid1=mid[4000:]
        mid3=sum(mid1)/float(len(mid1))
        result.append(mid3)

    return result
#    

# N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_game_compare1(N,r_1,p,c,w,u,M):
    g=initiate_graph(N,r_1,p)
    g=calculate_average_fitness(g,[[1,0],[1+c,c]],w)
    i=1
    result=[]
    while i<M:
        cooper=nx.get_node_attributes(g,'strategy')
        aver_cooper=sum(cooper.values())/float(g.order())
        result.append(aver_cooper)
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
    return result
#    
# N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_game_compare2(N,r_1,p,c,w,u,M):
    g=initiate_graph(N,r_1,p)
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
    return [result,result2]



