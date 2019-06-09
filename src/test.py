import numpy as np
import pandas as pd
from numpy.linalg import norm

a = np.array([1,2])
b = np.array([[1,1],[1,1]])

c = np.ones(a.size)

print(np.dot(a, c))

l1 = norm(a, 1)

print(l1)