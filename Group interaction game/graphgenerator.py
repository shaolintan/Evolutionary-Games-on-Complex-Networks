import networkx as nx
import numpy as np
import random 
import operator as op
import GraphMeasure as gm
import math



#-------------------------generate special graphs------------------------------

# generate graph with high heat heterogeneity with fixed degree sequence dic.
def heat_heter_graph(degseq):
    N=len(degseq)
    degseq=sorted(degseq,key=op.itemgetter(1))
    if N==0:
        g=nx.Graph()
        return g
    elif degseq[0][1]<0 or degseq[0][1]>N-1: 
        raise Exception
    else:
        g=nx.Graph()
        for i in range(degseq[0][1]):
            g.add_edges_from([(degseq[0][0],degseq[N-i-1][0])])
            degseq[N-i-1][1]=degseq[N-i-1][1]-1
        del degseq[0]
        g1=nx.compose(g,heat_heter_graph(degseq))
        return g1

#translate a degree list into a degree dictionary
def list_to_dic(deg):
    N=len(deg)
    dic=[]
    for i in range(N):
        dic.append([i,deg[i]])
    return dic

# generate graph with degree sequence by different means and compare their heat heterogeineity
def heter_of_seqgraph(K,M):
    var1=[]
    var2=[]
    gseq1=[]
    gseq2=[]
    num1=np.zeros(M)
    num2=np.zeros(M)
    i=0
    while i<M:
        seq=[]
        for j in range(K):
            a=random.randint(1,K-1)
            seq.append(a)
        dic=list_to_dic(seq)
        try:
            g1=heat_heter_graph(dic)
            if not(nx.is_connected(g1)):
                num1[i]=1
        except:
            num1[i]=2
        try:
            g2=nx.random_degree_sequence_graph(seq)
            if not(nx.is_connected(g2)):
                num2[i]=1
        except(nx.NetworkXUnfeasible, nx.NetworkXError):
            continue
        temp1=gm.Temp_var(g1)
        temp2=gm.Temp_var(g2)
        var1.append(temp1)
        var2.append(temp2)
        gseq1.append(g1)
        gseq2.append(g2)
        i+=1
    return (var1,var2,num1,num2,gseq1,gseq2)
        

# generate regular graph of size N and average M edges
def regular_graph(N,M):
    g=nx.Graph()
    for i in range(N):
        for j in range(M):
            k=(i+j+1)%N
            g.add_edge(i,k)
    return g


#generate graphs with high heat heterogeneity and pointed average degree and size 
def heter_graph(N,d):
    m=(3+math.sqrt((9-4*(2-d)*N)))/float(2)
    m=math.ceil(m)
    g=nx.gnm_random_graph(m,N*d/2-N+m)
    nodes=g.nodes()
    i=m
    while i<N:
        samp=random.sample(nodes,1)
        g.add_edge(i,samp[0])
        i+=1
    return g

# generate graph with degree sequence by different means and compare their heat heterogeineity
def heter_of_graphs(N,d,M):
    var1=0
    var2=0
    var3=0
    i=0
    while i<M:
        g1=heter_graph(N,d)
        g2=nx.gnp_random_graph(N,d/float(N-1))
        g3=nx.gnm_random_graph(N,d*N/2)
        temp1=gm.Temp_var(g1)
        temp2=gm.Temp_var(g2)
        temp3=gm.Temp_var(g3)
        var1=var1+temp1
        var2=var2+temp2
        var3=var3+temp3
        i+=1
    var1=var1/float(M)
    var2=var2/float(M)
    var3=var3/float(M)
    return (var1,var2,var3)

# compute the average degree for different average degree
def heat_vs_deg(N,M):
    var1=[]
    var2=[]
    var3=[]
    d=[]
    d1=2
    while d1<6:
        d1+=0.2
        res=heter_of_graphs(N,d1,M)
        var1.append(res[0])
        var2.append(res[1])
        var3.append(res[2])
        d.append(d1)   
    return (var1,var2,var3,d)
    
    
