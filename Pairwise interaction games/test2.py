import evolDynamic as ed
import networkx as nx
import GraphMeasure as gm


g=nx.read_edgelist('gmaxfit1015.edgelist')
temp=gm.Temp(g)

result=ed.record_BD_random_spreading(g,1.5,1000)

print result
print temp







