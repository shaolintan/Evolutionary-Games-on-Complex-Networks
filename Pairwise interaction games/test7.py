import networkx as nx
import numpy as np
import random as rd
import operator as op
import graphgenerator as gg




var1=[]
gseq1=[]
num1=np.zeros(12)
for i in range(12):
    seq=[]
    for j in range(10):
        a=rd.randint(1,9)
        seq.append(a)
    dic=gg.list_to_dic(seq)
    try:
        g1=gg.heat_heter_graph(dic)
        if not(nx.is_connected(g1)):
           num1[i]=1
    except:
        num1[i]=2
        g1=nx.complete_graph(10)
    temp1=gm.Temp_var(g1)
    var1.append(temp1)
