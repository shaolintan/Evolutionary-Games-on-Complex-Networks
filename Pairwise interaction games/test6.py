import evolDynamic as ed
import numpy as np

data=np.array([1,2,3,4])
index=np.array([1,2,1,2])
result=ed.data_average(data,index)
print result[0]

print result[1]

