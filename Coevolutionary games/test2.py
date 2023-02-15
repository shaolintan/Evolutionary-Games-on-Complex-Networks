import resize_graph as rz

import networkx as nx



g=nx.read_gml('evol_graph.gml')

g1=rz.resize_geo_graph(g,0.001)

nx.write_gml(g1,'resize_graph.gml')


