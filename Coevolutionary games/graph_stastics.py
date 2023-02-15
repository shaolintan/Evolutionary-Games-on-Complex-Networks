import networkx as nx

# return the largest connected component, and degree range (x),
# degree distribution (y), degree assortativity, trasitivity, and clustering

def net_stastics(g):
    H=nx.connected_component_subgraphs(g)[0]
    degree=nx.degree_histogram(H)
    deg_ran=range(len(degree))
    deg_dis=[z/float(sum(degree)) for z in degree]
    deg_assort=nx.degree_assortativity_coefficient(H)
    tran=nx.transitivity(H)
    clust=nx.average_clustering(H)
    return (H,deg_ran,deg_dis,deg_assort,tran,clust)
