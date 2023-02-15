import networkx as nx
import evolDynamic as ed
import numpy as np
import matplotlib.pyplot as plt



g=nx.barabasi_albert_graph(1000,3)
g2=nx.newman_watts_strogatz_graph(1000,3,0.4)
g3=nx.random_regular_graph(6,1000)
g4=nx.powerlaw_cluster_graph(1000,3,0.4)
mutant=ed.mutated_death_birth_simulation(g,2,40000,0.01)
mutant2=ed.mutated_death_birth_simulation(g2,2,40000,0.01)
mutant3=ed.mutated_death_birth_simulation(g3,2,40000,0.01)
mutant4=ed.mutated_death_birth_simulation(g4,2,40000,0.01)
mutant=np.array(mutant)
mutant2=np.array(mutant2)
mutant3=np.array(mutant3)
mutant3=np.array(mutant3)
aver=np.mean(mutant[-500:])
aver2=np.mean(mutant2[-500:])
aver3=np.mean(mutant3[-500:])
aver4=np.mean(mutant4[-500:])

print aver,aver2,aver3,aver4
plt.plot(mutant)
plt.plot(mutant2)
plt.plot(mutant3)
plt.plot(mutant4)
plt.show()




