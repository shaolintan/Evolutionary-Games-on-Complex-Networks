import networkx as nx
import GraphMeasure as gm
import evolDynamic as ed

g1=nx.read_edgelist('gmaxfit1015.edgelist')
g2=nx.complete_graph(10)
g3=nx.cycle_graph(10)
g4=nx.star_graph(9)
g1=gm.to_weighted(g1)
g2=gm.to_weighted(g2)
g3=gm.to_weighted(g3)
g4=gm.to_weighted(g4)
temp1=g1.in_degree(weight='weight')
temp2=g2.in_degree(weight='weight')
temp3=g3.in_degree(weight='weight')
temp4=g4.in_degree(weight='weight')
inv1=gm.inverse_temp(g1)
inv2=gm.inverse_temp(g2)
inv3=gm.inverse_temp(g3)
inv4=gm.inverse_temp(g4)


print temp1, temp2, temp3,temp4
print inv1,inv2,inv3,inv4


