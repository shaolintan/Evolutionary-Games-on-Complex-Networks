import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import GraphMeasure as gm

g=nx.read_gml('as-22july06.gml')
temp=gm.Temp(g)
result=np.histogram(temp,bins=200)
freq=[]
for i in range(200):
    freq.append(sum(result[0][i:])/float(2000))
b=result[1][:200]
plt.plot(b,freq)
plt.show()
