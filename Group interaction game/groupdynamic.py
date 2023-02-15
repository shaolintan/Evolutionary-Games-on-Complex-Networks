import numpy as np
import random as rd
import evolDynamic as ed
import math
import networkx as nx

# calculate the accumerate fitness of each individual
def group_calculate_fitness(g,pm,w):
    zeros={}
    ones={}
    for d1 in g.nodes_iter():
        zeros[d1]=0
        ones[d1]=0
        zeros[d1]=zeros[d1]+1-g.node[d1]['type']
        ones[d1]=ones[d1]+g.node[d1]['type']
        for d2 in g.neighbors_iter(d1):
            zeros[d1]=zeros[d1]+1-g.node[d2]['type']
            ones[d1]=ones[d1]+g.node[d2]['type']
    payoff={}
    for d in g.nodes_iter():
        payoff[d]=0
    for d1 in g.nodes_iter():
        payoff[d1]=payoff[d1]+(g.node[d1]['type']*(pm[0][0]*ones[d1]+\
                   pm[0][1]*zeros[d1])+(1-g.node[d1]['type'])*(pm[1][0]*ones[d1]+\
                   pm[1][1]*zeros[d1]))/float(ones[d1]+zeros[d1])
        for d2 in g.neighbors_iter(d1):
            payoff[d2]=payoff[d2]+(g.node[d2]['type']*(pm[0][0]*ones[d1]+\
                   pm[0][1]*zeros[d1])+(1-g.node[d2]['type'])*(pm[1][0]*ones[d1]+\
                   pm[1][1]*zeros[d1]))/float(ones[d1]+zeros[d1])
    for d in g.nodes_iter():
        payoff[d]=math.exp(w*payoff[d])
    nx.set_node_attributes(g,'fitness',payoff)
    return g

# calculate the accumurate fitness of each individual
def pairwise_calculate_fitness(g,pm,w):
    payoff={}
    for d1 in g.nodes_iter():
        payoff[d1]=0
        for d2 in g.neighbors_iter(d1):
            mid1=np.dot([g.node[d1]['type'],1-g.node[d1]['type']],pm)
            payoff[d1]+=np.dot(mid1,[g.node[d2]['type'],1-g.node[d2]['type']])
        payoff[d1]=1-w+w*payoff[d1]
    nx.set_node_attributes(g,'fitness',payoff)
    return g

# select the individual for reproducation with a probability proportional to their fitness
def BD_selected_birth(g):
    nodes=g.nodes()
    totalfit=0
    for d1 in nodes:
        totalfit=totalfit+g.node[d1]['fitness']
    rand=rd.random()
    start=0
    end=g.node[nodes[0]]['fitness']/float(totalfit)
    for i in range(len(nodes)):
        if (rand>=start) and (rand<end):
            birth=nodes[i]    # birth is node
            break
        start=end
        end=end+g.node[nodes[i+1]]['fitness']/float(totalfit)
    return birth

# select the individual for reproduction in the death-birth process
def DB_selected_birth(g,death):
    totalfit=0
    for d1 in g.neighbors_iter(death):
        totalfit=totalfit+g.node[d1]['fitness']
    nodes=g.neighbors(d1)
    rand=rd.random()
    start=0
    end=g.node[nodes[0]]['fitness']/float(totalfit)
    for i in range(len(nodes)):
        if (rand>=start) and (rand<end):
            birth=nodes[i]    # birth is node
            break
        start=end
        end=end+g.node[nodes[i+1]]['fitness']/float(totalfit)
    return birth

# simulate the evolutionary game with group BD updating
def group_BD_game_evol(g,game,w,nbunch,u,N):
    g=ed.add_dynamic_attributes(g,nbunch)
    result=[]
    i=0
    while i<N:
        g=group_calculate_fitness(g,game,w)
        birth=BD_selected_birth(g)
        death=ed.BD_random_death(g,birth)
        p=rd.random()
        if p>=u:
            g.node[death[0]]['type']=g.node[birth]['type']
        else:
            q=rd.random()
            if q>=0.5: g.node[death[0]]['type']=0
            else: g.node[death[0]]['type']=1
        cooper=ed.test_nodes_type(g)
        result.append(cooper)
        i+=1
    return result
        
# simulate the evolutionary game with pairwise BD updating
def pairwise_BD_game_evol(g,game,w,nbunch,u,N):
    g=ed.add_dynamic_attributes(g,nbunch)
    result=[]
    i=0
    while i<N:
        g=pairwise_calculate_fitness(g,game,w)
        birth=BD_selected_birth(g)
        death=ed.BD_random_death(g,birth)
        p=rd.random()
        if p>=u:
            g.node[death[0]]['type']=g.node[birth]['type']
        else:
            q=rd.random()
            if q>=0.5: g.node[death[0]]['type']=0
            else: g.node[death[0]]['type']=1
        cooper=ed.test_nodes_type(g)
        result.append(cooper)
        i+=1
    return result





        
