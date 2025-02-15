"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt

c = PCircuit(4)


#c.J = np.array([[0,-1,2,0,0],[-1,0,2,0,0],[2,2,0,-1,2],[0,0,-1,0,2],[0,0,2,2,0]])
#c.h = np.array([1,1,-1,1,-2])


c.J = np.array([[0,-1,2,0],[-1,0,2,0],[2,2,0,1],[0,0,1,0]])
c.h = np.array([1,1,-2,0])

solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input, output = solver.solve(c)

histplot(output)


np.set_printoptions(threshold=np.inf)
print(output)


#3d Histogram plot for the p-bit
plot3d(output, A=[0,1], B=[2,3])

vin_vout(input, output, p_bit=2)
