import networkx as nx
import random as rd


def resize_geo_graph(g,r):
    nodes=g.nodes()
    g1=nx.Graph()
    while nodes:
        u=nodes.pop()
        for v in nodes:
            if abs(g.node[u]['pos']-g.node[v]['pos'])<=r:
                g1.add_edge(u,v)
    return g1

