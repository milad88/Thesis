import numpy as np

a = np.ones([4])
b = np.zeros([4])
c = np.array([a,b])
for r in c:
    r = r *2

print(c)