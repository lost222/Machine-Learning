import time
import numpy as np

Y = np.load("Y.npy")

start = time.clock()
for i in range(1000):
    aMax = np.max(Y)
t_max = time.clock()
for i in range(1000):
    a_min = np.min(Y)
t_min = time.clock()
for i in range(1000):
    a_mean = np.mean(Y)
t_mean = time.clock()

print("time to get max ", t_max - start)
print("time to get min ", t_min - t_max)
print("time to get mean", t_mean - t_min)