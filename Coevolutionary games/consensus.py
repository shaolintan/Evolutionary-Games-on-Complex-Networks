# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:23:34 2019

@author: sean
"""

import networkx as nx
import random as rd
import matplotlib.pyplot as plt


def consensus(g,delta,t):
    for d in g.nodes():
        g.node[d]['state']=rd.random()
        g.node[d]['evolution']=[g.node[d]['state']]
    i=0
    while i<=t:
        state=nx.get_node_attributes(g,'state')
        for d1 in g.nodes():
            for d2 in nx.all_neighbors(g,d1):
                g.node[d1]['state']=g.node[d1]['state']-2*delta*(state[d1]-state[d2])
            g.node[d1]['evolution'].append(g.node[d1]['state'])    
        i=i+1
    return g

g=nx.cycle_graph(10)
g=consensus(g,0.02,100)

for d in g.nodes():
    plt.plot(g.node[d]['evolution'])