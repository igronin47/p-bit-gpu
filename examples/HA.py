"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt

c = PCircuit(5)



#c.J = np.array([[0,-1,1,2],[-1,0,1,2],[1,1,0,-2],[2,2,-2,0]])
#c.h = np.array([1,1,-1,-2])


#c.J = np.array([[0,-3,2,2,0,0,2],[-3,0,2,2,0,0,2],[2,2,0,0,-1,2,0],[2,2,0,0,-1,0,0],[0,0,-1,-1,0,2,0],[0,0,2,0,2,0,0],[2,2,0,0,0,0,0]])
#c.h = np.array([1,1,3,-2,1,-2,-2])


#HA using Majority (Reduced)
c.J = np.array([[0,0,3,-3,2],[0,0,2,2,0],[3,2,0,1,-2],[-3,2,1,0,2],[2,0,-2,2,0]])
c.h = np.array([0,2,-2,-2,0])


solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)

#np.set_printoptions(threshold=np.inf)
#print(output)

histplot(output)


#3d Histogram plot for the p-bit
#plot3d(output, A=[0,1,2,3], B=[4,5,6])


#vin_vout(input, output, p_bit=6)