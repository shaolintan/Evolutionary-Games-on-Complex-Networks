import numpy as np
import random as rd
import networkx as nx

# Initiate a random geometric graph with attributes (strategy=random, pos=random in [0 1], birth_date=1, ancestor_type=[1,N])
# N is the number of nodes, and r is the radius
def initiate_graph(N,r):
    g=nx.Graph()
    g.add_nodes_from(range(N))
    for d in g.nodes():
        g.node[d]['strategy']=rd.randint(0,1)
        g.node[d]['pos']=rd.random()
        g.node[d]['birth_date']=1
        g.node[d]['anc_type']=d
    nodes=g.nodes()
    while nodes:
        u=nodes.pop()
        for v in nodes:
            if abs(g.node[u]['pos']-g.node[v]['pos'])<=r:
                g.add_edge(u,v)
    return g

# calculate the accumurate fitness of each individual
# g is the graph, pm is the game, w is the selection intensity
def calculate_fitness(g,pm,w):
    payoff={}
    for d1 in g.nodes_iter():
        payoff[d1]=0
        for d2 in g.neighbors_iter(d1):
            mid1=np.dot([g.node[d1]['strategy'],1-g.node[d1]['strategy']],pm)
            payoff[d1]+=np.dot(mid1,[g.node[d2]['strategy'],1-g.node[d2]['strategy']])
        payoff[d1]=1-w+w*payoff[d1]
    nx.set_node_attributes(g,'fitness',payoff)
    return g



# calculate the average fitness of each individual
# g is the graph, pm is the game, w is the selection intensity
def calculate_average_fitness(g,pm,w):
    payoff={}
    for d1 in g.nodes_iter():
        payoff[d1]=0
        for d2 in g.neighbors_iter(d1):
            mid1=np.dot([g.node[d1]['strategy'],1-g.node[d1]['strategy']],pm)
            payoff[d1]+=np.dot(mid1,[g.node[d2]['strategy'],1-g.node[d2]['strategy']])
        if payoff[d1]!=0:
            payoff[d1]=payoff[d1]/g.degree()[d1]
        payoff[d1]=1-w+w*payoff[d1]
    nx.set_node_attributes(g,'fitness',payoff)
    return g


# selecte the individual for reproducation with a probability proportional to their fitness
def selected_birth(g):
    nodes=g.nodes()
    totalfit=0
    for d1 in nodes:
        totalfit=totalfit+g.node[d1]['fitness']
    rand=rd.random()
    start=0
    end=g.node[nodes[0]]['fitness']/float(totalfit)
    for i in range(len(nodes)):
        if rand<end:
            birth=nodes[i]    # birth is node
            break
        start=end
        end=end+g.node[nodes[i+1]]['fitness']/float(totalfit)
    return birth



# test whether all the nodes have the same type
def test_nodes_type(g):
    result=0
    for d1 in g.nodes_iter():
        result+=g.node[d1]['strategy']
    result=float(result)/g.order()
    return result


# The evolutionary process
# N,r is used to initialize graph, c is the defection tempation, w is the selection intensity, u is the mutation rate, M is the steps
def evol_game(N,r_1,r_2,c,w,u,M):
    g=initiate_graph(N,r_1)
    g_r=nx.Graph()
    g_r.add_nodes_from(g.nodes(data=True))
    i=1
    result=[]
    while i<M:
            g=calculate_average_fitness(g,[[1,0],[1+c,c]],w)
            nodes=nx.nodes(g)
            death=rd.sample(nodes,1)
            g.remove_node(death[0])
            g_r.remove_node(death[0])
            nodes=g.nodes()
            birth=selected_birth(g)
            k=i+N-1
            pos_k=g.node[birth]['pos']-r_2+2*r_2*rd.random()
            if pos_k>1:
                pos_k=1
            elif pos_k<0:
                pos_k=0
            g.add_node(k,pos=pos_k,birth_date=i,anc_type=g.node[birth]['anc_type'])
            p=rd.random()
            if p>=u:
                g.node[k]['strategy']=g.node[birth]['strategy']
            else:
                q=rd.random()
                if q>=0.5:
                    g.node[k]['strategy']=0
                else:
                    g.node[k]['strategy']=1
            for d in nodes:
                if abs(g.node[d]['pos']-g.node[k]['pos'])<=r_1:
                    g.add_edge(d,k)
            g_r.add_node(k,pos=pos_k,birth_date=i,anc_type=g.node[birth]['anc_type'],strategy=g.node[k]['strategy'])
            g_r.add_edge(birth,k)
            cooper=test_nodes_type(g)
            result.append(cooper)
            i+=1
    eout=[result,g,g_r]
    return eout





        

        


  

  

  


