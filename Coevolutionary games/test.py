import evol_network as en

import networkx as nx

import matplotlib.pyplot as plt

import numpy as np


rslt=en.evol_game(1000,0.15,0.02,0.2,0.1,0.01,110000)
#print rslt[0]


#np.save('cooper_045022101_27.npy',rslt[0])

#nx.write_gml(rslt[1],'evol_graph_27.gml')



#nx.write_gml(rslt[2],'relation_graph_27.gml')

plt.plot(rslt[0],'r-')




# note that from 19 to 20, the parameter increases from 0.02 to 0.03
# note that from 22 to 23, the parameter increases from 0.05 to 0.07



