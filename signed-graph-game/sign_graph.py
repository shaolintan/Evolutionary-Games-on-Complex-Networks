import networkx as nx
import random as rd
import numpy as np
import math
import graphgenerator as gg

# initiate the nodes with given value
def add_dynamic_attributes(g,value):
    a={}
    i=0
    for d1 in g.nodes():
        a[d1]=value[i]
        i=i+1
    nx.set_node_attributes(g,'prob',a)
    return g

# update the nodes' dynamical attributes adaptively
def one_step_updating(G,d,bina):
    s={}
    for d1 in G.nodes():
        s[d1]=G.node[d1]['prob']
        for d2 in G.neighbors_iter(d1):
            s[d1]=s[d1]+d*G[d1][d2]['weight']*(4*G.node[d2]['prob']-2)/float(G.degree(d1))+bina*0.1*d*(rd.random()-0.5)
    for d1 in G.nodes():
        if s[d1]>1:
            G.node[d1]['prob']=1
        elif s[d1]<0:
            G.node[d1]['prob']=0
        else:
            G.node[d1]['prob']=s[d1]
    return G

# simulate the evolutionary process 
def evolution(G,d,M,bina):
    path={}
    for d1 in G.nodes():
        path[d1]=[G.node[d1]['prob']]
    for i in range(M):
        G=one_step_updating(G,d,bina)
        for d1 in G.nodes():
            path[d1].append(G.node[d1]['prob'])
    return (G,path)
    
# partite the nodes into three parts
def partition(G):
    part_p=[]
    part_n=[]
    part_m=[]
    for d3 in G.nodes_iter():
        if G.node[d3]['prob']==1:
            part_p.append(d3)
        elif G.node[d3]['prob']==0:
            part_n.append(d3)
        else:
            part_m.append(d3)
    return (part_p,part_n,part_m)

# re-simulate the evolutionary process
def re_evolution(G,d,K):
    part=partition(G)
    N=len(part[2])
    if N>0:
        for d3 in part[2]:
            G.node[d3]['prob']=rd.random()
        G=evolution(G,d,K)
    return G

# the whole algorithm to determine the unbalanced distance
def evo_alg(G,d,M,R):
    N=G.order()
    dis=np.random.rand(N)
    G=add_dynamic_attributes(G,dis)
    G=evolution(G,d,M)
    for i in range(R):
        G=re_evolution(G,d,M)
    return G
         
# computate the balanced level of the partition
def balanced_level(G):
    level_1=0
    level_2=0
    for d1 in G.nodes():
        for d2 in G.neighbors_iter(d1):
            a=(-1+2*G.node[d1]['prob'])*G[d1][d2]['weight']*(-1+2*G.node[d2]['prob'])
            if a>0:
                level_1+=1
            else:
                level_2+=1
    return (level_1,level_2)

# computate the balanced level of the partition
def balanced_level_2(G):
    level_1=0
    level_2=0
    part=partition(G)
    for d1 in part[0] or part[1]:
        for d2 in G.neighbors_iter(d1):
            a=(-1+2*G.node[d1]['prob'])*G[d1][d2]['weight']*(-1+2*G.node[d2]['prob'])
            if a>0:
                level_1+=1
            else:
                level_2+=1
    return (level_1,level_2)


############################### evolutinary game method #############################
# add dynamic distributions
def add_dynamic_attributes_2(g):
    a={}
    p={}
    for d in g.nodes():
        a[d]=1
        p[d]=0
    nx.set_node_attributes(g,'strategy',a)
    nx.set_node_attributes(g,'payoff',p)
    return g



# derive the nodes set whose payoff is less than 0
def departe_graph(g):
    comp_1=set()
    comp_2=set()
    comp_3=set()
    total_payoff=0
    edge_number=0
    for d1 in g.nodes():
        g.node[d1]['payoff']=0
        for d2 in g.neighbors_iter(d1):
            g.node[d1]['payoff']=g.node[d1]['payoff']+g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
            edge_number+=1
        total_payoff+=g.node[d1]['payoff']
        if g.node[d1]['payoff']<0:
            comp_1.add(d1)
        elif g.node[d1]['payoff']==0:
            comp_2.add(d1)
        else:
            comp_3.add(d1)
    return (comp_1,comp_2,comp_3,edge_number,total_payoff,g)

# derive the nodes set whose payoff is less than 0 in directed graphs
def directed_departe_graph(g):
    comp_1=set()
    comp_2=set()
    comp_3=set()
    total_payoff=0
    edge_number=0
    for d in g.nodes():
        g.node[d]['payoff']=0
    for d1 in g.nodes():
        for d2 in g.neighbors_iter(d1):
            g.node[d1]['payoff']=g.node[d1]['payoff']+g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
            g.node[d2]['payoff']=g.node[d2]['payoff']+g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
            edge_number+=1
    for d1 in g.nodes():
        total_payoff+=g.node[d1]['payoff']
        if g.node[d1]['payoff']<0:
            comp_1.add(d1)
        elif g.node[d1]['payoff']==0:
            comp_2.add(d1)
        else:
            comp_3.add(d1)
    return (comp_1,comp_2,comp_3,edge_number,total_payoff,g)                


# state updating
def state_updating(g,comp_1,comp_2,comp_3,total_payoff):
    while len(comp_1)>0:
        d1=comp_1.pop()
        comp_3.add(d1)
        g.node[d1]['strategy']=-g.node[d1]['strategy']
        g.node[d1]['payoff']=-g.node[d1]['payoff']
        for d2 in g.neighbors_iter(d1):
            g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
            if g.node[d2]['payoff']<0:
                comp_1.add(d2)
                comp_2.discard(d2)
                comp_3.discard(d2)
            elif g.node[d2]['payoff']==0:
                comp_2.add(d2)
                comp_1.discard(d2)
                comp_3.discard(d2)
            else:
                comp_3.add(d2)
                comp_1.discard(d2)
                comp_2.discard(d2)
        total_payoff=total_payoff+4*g.node[d1]['payoff']
    return (g,comp_1,comp_2,comp_3,total_payoff)

# repeate the updating process after resign the state of those nodes whose payoff is zero for T times
def evo_alg_2(g,T):
    g=add_dynamic_attributes_2(g)
    payoff_seq=[]
    for j in range(T):
        mid1=departe_graph(g)
        payoff_seq.append(mid1[4])
        mid2=state_updating(g,mid1[0],mid1[1],mid1[2],mid1[4])
        payoff_seq.append(mid2[4])
        g=mid2[0]
        if len(mid2[2])==0 or (i==T-1):
            break
        for d1 in mid2[2]:
            g.node[d1]['strategy']=rd.choice([1,-1])
    return (g,mid1[3],payoff_seq,mid2[1],mid2[2])

# repeate the updating process after resign the initial state of all nodes for T times
def evo_alg_3(g,T):
    payoff_seq=[]
    for i in range(T):
        g=add_dynamic_attributes_2(g)
        mid1=departe_graph(g)
        payoff_seq.append(mid1[4])
        mid2=state_updating(g,mid1[0],mid1[1],mid1[2],mid1[4])
        payoff_seq.append(mid2[4])
        g=mid2[0]
    return (g,mid1[3],payoff_seq,mid2[1],mid2[2])


# one step updating of the state
def one_step_updating_2(g,T):
    for d1 in g.nodes():
        if g.node[d1]['payoff']<0:
            g.node[d1]['strategy']=-g.node[d1]['strategy']
            g.node[d1]['payoff']=-g.node[d1]['payoff']
            for d2 in g.neighbors_iter(d1):
                g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
        else:
            p=rd.random()
            if p<0.5*math.exp(-float(g.node[d1]['payoff'])/T):
                g.node[d1]['strategy']=-g.node[d1]['strategy']
                g.node[d1]['payoff']=-g.node[d1]['payoff']
                for d2 in g.neighbors_iter(d1):
                    g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
    total_payoff=0
    for d1 in g.nodes():
        total_payoff+=g.node[d1]['payoff']
    return (g,total_payoff)



# one step updating of the state
def const_one_step_updating_2(g,m):
    for d1 in g.nodes():
        if g.node[d1]['payoff']<0:
            g.node[d1]['strategy']=-g.node[d1]['strategy']
            g.node[d1]['payoff']=-g.node[d1]['payoff']
            for d2 in g.neighbors_iter(d1):
                g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
        else:
            p=rd.random()
            if p<m:
                g.node[d1]['strategy']=-g.node[d1]['strategy']
                g.node[d1]['payoff']=-g.node[d1]['payoff']
                for d2 in g.neighbors_iter(d1):
                    g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
    total_payoff=0
    for d1 in g.nodes():
        total_payoff+=g.node[d1]['payoff']
    return (g,total_payoff)

# one step updating of the state for directed graphs
def directed_one_step_updating_2(g,T):
    for d1 in g.nodes():
        if g.node[d1]['payoff']<0:
            g.node[d1]['strategy']=-g.node[d1]['strategy']
            g.node[d1]['payoff']=-g.node[d1]['payoff']
            for d2 in g.neighbors(d1):
                g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
            for d2 in g.predecessors(d1):
                g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d2][d1]['weight']
        else:
            p=rd.random()
            if p<0.5*math.exp(-float(g.node[d1]['payoff'])/T):
                g.node[d1]['strategy']=-g.node[d1]['strategy']
                g.node[d1]['payoff']=-g.node[d1]['payoff']
                for d2 in g.neighbors_iter(d1):
                    g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']
                for d2 in g.predecessors(d1):
                    g.node[d2]['payoff']=g.node[d2]['payoff']+2*g.node[d1]['strategy']*g.node[d2]['strategy']*g[d2][d1]['weight']
    total_payoff=0
    for d1 in g.nodes():
        total_payoff+=g.node[d1]['payoff']
    return (g,total_payoff)

# repeat the above process
def evo_alg_4(g,T,a,N,K):
    g=add_dynamic_attributes_2(g)
    payoff_seq=[]
    conf_seq=[]
    mid1=departe_graph(g)
    g=mid1[5]
    payoff_seq.append(mid1[4])
    for i in range(K):
        T_k=pow(a,i)*T
        for j in range(N):
            mid2=one_step_updating_2(g,T_k)
            g=mid2[0]
            conf=0
            for d1 in g.nodes():
                for d2 in g.neighbors_iter(d1):
                    if g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']<0:
                        conf+=abs(0.5*g[d1][d2]['weight'])
            payoff_seq.append(mid2[1])
            conf_seq.append(conf)
    return (g,conf_seq,payoff_seq)


# repeat the above process
def heter_mutat_evo_alg_4(g,T,a,N,K):
    g=add_dynamic_attributes_2(g)
    payoff_seq=[]
    conf_seq=[]
    mid1=departe_graph(g)
    g=mid1[5]
    payoff_seq.append(mid1[4])
    for i in range(K):
        T_k=T
        for j in range(N):
            mid2=one_step_updating_2(g,T_k)
            g=mid2[0]
            conf=0
            for d1 in g.nodes():
                for d2 in g.neighbors_iter(d1):
                    if g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']<0:
                        conf+=abs(0.5*g[d1][d2]['weight'])
            payoff_seq.append(mid2[1])
            conf_seq.append(conf)
    return (g,conf_seq,payoff_seq)


# repeat the above process for directed signed graphs
def directed_evo_alg_4(g,T,a,N,K):
    g=add_dynamic_attributes_2(g)
    payoff_seq=[]
    conf_seq=[]
    mid1=directed_departe_graph(g)
    g=mid1[5]
    payoff_seq.append(mid1[4])
    for i in range(K):
        T_k=pow(a,i)*T
        for j in range(N):
            mid2=directed_one_step_updating_2(g,T_k)
            g=mid2[0]
            conf=0
            for d1 in g.nodes():
                for d2 in g.neighbors(d1):
                    if g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']<0:
                        conf+=1
            payoff_seq.append(mid2[1])
            conf_seq.append(conf)
    return (g,conf_seq,payoff_seq)

# compute the structural conflicts of a series signed graphs
def struc_conf(N1,p1,p2,K1,T,a,N2,K2):
    real_conf=[]
    derived_conf=[]
    for s in range(0,K1,5):
        rslt=gg.designed_weighted_signed_networks(N1,s,p1,p2)
        g=rslt[0]
        g=add_dynamic_attributes_2(g)
        size=g.size()
        mid1=departe_graph(g)
        g=mid1[5]
        for i in range(K2):
            T_k=pow(a,i)*T
            for j in range(N2):
                mid2=one_step_updating_2(g,T_k)
                g=mid2[0]
        conf=0  
        for d1 in g.nodes():
            for d2 in g.neighbors_iter(d1):
                if g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']<0:
                    conf+=abs(0.5*g[d1][d2]['weight'])
        real_conf.append(rslt[1])
        derived_conf.append(conf)
    return (real_conf,derived_conf)
    
# compute the structural conflicts of a series directed signed graphs
def directed_struc_conf(N1,p1,p2,K1,T,a,N2,K2):
    real_conf=[]
    derived_conf=[]
    for s in range(0,K1,5):
        rslt=gg.designed_directed_signed_networks(N1,s,p1,p2)
        g=rslt[0]
        g=add_dynamic_attributes_2(g)
        size=g.size()
        mid1=directed_departe_graph(g)
        g=mid1[5]
        for i in range(K2):
            T_k=pow(a,i)*T
            for j in range(N2):
                mid2=directed_one_step_updating_2(g,T_k)
                g=mid2[0]
        conf=0  
        for d1 in g.nodes():
            for d2 in g.neighbors_iter(d1):
                if g.node[d1]['strategy']*g.node[d2]['strategy']*g[d1][d2]['weight']<0:
                    conf+=abs(g[d1][d2]['weight'])
        real_conf.append(rslt[2])
        derived_conf.append(conf)
    return (real_conf,derived_conf)
     



    
