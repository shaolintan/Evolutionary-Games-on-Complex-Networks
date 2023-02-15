import networkx as nx
import evolDynamic as ed

g1=nx.Graph()
g1.add_edges_from([(0,1),(1,2),(1,3),(1,4),(1,5),(1,6),(3,4),(3,5),(3,6),(4,5),(4,6),(5,6)])
f1_BD=ed.BD_average_fixation(g1,1.5,3000)
f1_DB=ed.DB_average_fixation(g1,1.5,3000)
print f1_BD, f1_DB

