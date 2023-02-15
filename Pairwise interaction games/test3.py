import evolDynamic as ed
import numpy as np
import matplotlib.pyplot as plt

increase=ed.increase_mutant(1000,4,10000,0.01)

plt.plot(increase)
plt.show()
