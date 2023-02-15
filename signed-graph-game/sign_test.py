import networkx as nx
import log_linear_graph as llg
import numpy as np
import random as rd
import matplotlib.pyplot as plt
#import graphgenerator as gg


g=nx.read_weighted_edgelist('yeast_symm.txt')
#g=nx.Graph()
#g.add_edge(1,2,weight=-1)
#g.add_edge(2,3,weight=1)
#g.add_edge(3,4,weight=-1)
#g.add_edge(4,1,weight=1)

#g=sg.add_dynamic_attributes_2(g)
#g.node[1]['strategy']=-1
#g.node[2]['strategy']=-1
#g.node[3]['strategy']=1
#g.node[4]['strategy']=1

#rslt=sg.departe_graph(g)

#a=gg.designed_directed_signed_networks(300,20,0.05,0.003)
#g=a[0]
#rslt2=sg.one_step_updating_2(g,rslt[0],rslt[1],rslt[2],rslt[4])
#rslt=sg.directed_evo_alg_4(g,100,0.9,100,600)

#rslt=sg.directed_struc_conf(300,0.01,0.006,155,100,0.9,1000,60)
#g1=sg.evo_alg_2(g,3000,10)
#rslt2=sg.partition(g1)
# g2=sg.evo_alg(g,0.02,2000,5)
# rslt2=sg.balanced_level(g2)
#rslt1=sg.heter_mutat_evo_alg_4(g,1,0.9,600,100)
#rslt2=sg.heter_mutat_evo_alg_4(g,0.001,0.9,600,100)
#rslt1=sg.evo_alg_4(g,100,0.6,600,100)
#rslt2=sg.evo_alg_4(g,10,0.9,600,100)
rslt=llg.replicator_based_learning(g,0.005,20000)
a=rslt[1]
plt.plot(a)
plt.show()



