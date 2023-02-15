#import networkx as nx
#import log_linear_graph as llg
import sensor_coverage_problem as scp
import numpy as np
import random as rd
import matplotlib.pyplot as plt

a=[(1,1),(4,3),(5,3)]
u=[0.75,0.45]
rslt=scp.general_updating(a,u,1000)
plt.plot(rslt[0])
plt.show()
