import numpy as np
import networkx as nx

#-----------------------------death-birth process-----------------------------
# translate the graph into the death-birth matrix for fixation
def graphtrans(g):
    g=nx.adjacency_matrix(g)
    g=np.array(g)
    N=len(g)
    a=g.sum(0)
    b=np.empty([N,N])
    for i in range(N):
        for j in range(N):
            if i==j:
                b[i][j]=float(N-1)/N
            else:
                b[i][j]=float(g[j][i])/(N*a[i])
    return b

# calculate the fixation probability when the fitness is 1
def fix_vs_mutantset(g):
    g1=graphtrans(g)
    g1=np.transpose(g1)
    eigvl,eigvc=np.linalg.eig(g1)
    eigvl=list(eigvl)
    eig=eigvc[:,0]
    eig=eig/sum(eig)
    return eig



       
    
