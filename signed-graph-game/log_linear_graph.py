import networkx as nx
import random as rd
import numpy as np
import math
#import graphgenerator as gg


############################### Log-Linear Learning #############################
# add dynamic distributions
def add_dynamic_attributes(g):
    for d in g.nodes():
        g.nodes[d]['strategy']=1
        g.nodes[d]['payoff']=0
        g.nodes[d]['ind_set']=nx.maximal_independent_set(g,[d])
    total_payoff=0
    for d1 in g.nodes():
        for d2 in g.neighbors(d1):
            g.node[d1]['payoff']=g.node[d1]['payoff']+g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
        total_payoff+=g.node[d1]['payoff']
    g.graph['total_payoff']=total_payoff
    return g

#####################################################################
# one step asynchronous updating with log-linear learning
def one_step_updating(g,w):
    nodes=g.nodes()
    d=rd.sample(nodes,1)[0]
    p=rd.random()
    if p<float(1/(1+math.exp(2*w*g.node[d]['payoff']))):
        g.node[d]['strategy']=-g.node[d]['strategy']
        g.node[d]['payoff']=-g.node[d]['payoff']
        for d2 in g.neighbors(d):
            g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d]['strategy']*g.node[d2]['strategy']*g[d][d2]['weight']
        total_payoff=0
        for d1 in g.nodes():
            total_payoff+=g.node[d1]['payoff']
        g.graph['total_payoff']=total_payoff
    return g

# the learning process
def general_learning(g,T):
    g=add_dynamic_attributes(g)
    payoff_seq=[]
    payoff_seq.append(g.graph['total_payoff'])
    for t in range(T):
        w=math.log(t+1)/5
        g=one_step_updating(g,w)
        payoff_seq.append(g.graph['total_payoff'])
    return (g,payoff_seq)
#######################################################################


#######################################################################               
# one step updating, each time an independent maximal set is selected to update
def ind_one_step_updating(g,w):
    nodes=g.nodes()
    d=rd.sample(nodes,1)[0]
    p=rd.random()
    for d1 in g.nodes[d]['ind_set']:
        b=min(2*w*g.node[d1]['payoff'],500)
        if p<float(1/(1+math.exp(b))):
            g.node[d1]['strategy']=-g.node[d1]['strategy']
            g.node[d1]['payoff']=-g.node[d1]['payoff']
            for d2 in g.neighbors(d1):
                g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
    total_payoff=0
    for d3 in g.nodes():
        total_payoff+=g.node[d3]['payoff']
    g.graph['total_payoff']=total_payoff
    return g

# the learning process
def ind_general_learning(g,T):
    g=add_dynamic_attributes(g)
    payoff_seq=[]
    payoff_seq.append(g.graph['total_payoff'])
    for t in range(T):
        w=math.pow(math.log(t+1),0.6)
        g=ind_one_step_updating(g,w)
        payoff_seq.append(g.graph['total_payoff'])
    return (g,payoff_seq)
#######################################################################


#######################################################################
# add dynamic distributions including probability
def add_dynamic_attributes_2(g):
    for d in g.nodes():
        g.nodes[d]['strategy']=1
        g.nodes[d]['payoff']=0
        g.nodes[d]['bias']=0.5
    total_payoff=0
    for d1 in g.nodes():
        for d2 in g.neighbors(d1):
            g.node[d1]['payoff']=g.node[d1]['payoff']+g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
        total_payoff+=g.node[d1]['payoff']
    g.graph['total_payoff']=total_payoff
    return g

# one step updating of bias
def bias_updating(g,w):
    for d in g.nodes():
        aver_payoff=g.nodes[d]['bias']*math.exp(w*g.nodes[d]['strategy']*g.nodes[d]['payoff'])+(1-g.nodes[d]['bias'])*math.exp(-w*g.nodes[d]['strategy']*g.nodes[d]['payoff'])
        g.nodes[d]['bias']=g.nodes[d]['bias']*math.exp(w*g.nodes[d]['strategy']*g.nodes[d]['payoff'])/aver_payoff
    return g

# one step updating of state(strategy and payoff)
def state_updating(g):
    for d in g.nodes():
        p=rd.random()
        if p<g.nodes[d]['bias']:
            g.nodes[d]['strategy']=1
        else:
            g.nodes[d]['strategy']=-1
    total_payoff=0
    for d1 in g.nodes():
        g.node[d1]['payoff']=0
        for d2 in g.neighbors(d1):
            g.node[d1]['payoff']=g.node[d1]['payoff']+g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
        total_payoff+=g.node[d1]['payoff']
    g.graph['total_payoff']=total_payoff
    return g

# the learning process
def replicator_based_learning(g,w,T):
    g=add_dynamic_attributes_2(g)
    payoff_seq=[]
    payoff_seq.append(g.graph['total_payoff'])
    for t in range(T):
        g=bias_updating(g,w)
        g=state_updating(g)
        payoff_seq.append(g.graph['total_payoff'])
    return (g,payoff_seq)
########################################################################
