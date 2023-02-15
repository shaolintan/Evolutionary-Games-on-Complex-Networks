import numpy as np
import math


def random_walk(r,d):
    theta=np.random.uniform(0,2*math.pi)
    r[0]=r[0]+d*math.cos(theta)
    r[1]=r[1]+d*math.sin(theta)
    return r

def boundary(B,c):
    
    
    
    
