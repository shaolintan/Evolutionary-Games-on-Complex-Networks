import evolDynamic as ed
import networkx as nx


gmin=nx.read_pajek('gminf.pajek')

r4=ed.edges_vs_mutants(gmin,1000)

