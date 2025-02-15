"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout, plot3d
import numpy as np
import matplotlib.pyplot as plt



#c = PCircuit(5)

#c.J = np.array([[0,0,0,2,-2],[0,0,-2,2,2],[0,-2,0,2,2],[2,2,2,0,1],[-2,2,2,1,0]])
#c.h = np.array([-2,0,0,3,-3])


c = PCircuit(7)

#Matrix using Hamiltonian Equation
c.J = np.array([[0,-2,2,2,0,0,0],[-2,0,2,2,0,0,0],[2,2,0,0,-1,2,0],[2,2,0,0,-1,0,0],[0,0,-1,-1,0,2,0],[0,0,2,0,2,0,-1],[0,0,0,0,0,-1,0]])
c.h = np.array([0,0,3,-2,1,-2,0])


#reduced p bits using given XOR Gate matrix
#c.J = np.array([[0,0,2,-2,0,0],[0,0,2,2,-2,0],[2,2,0,1,2,0],[-2,2,1,0,2,0],[0,-2,2,2,0,-1],[0,0,0,0,-1,0]])
#c.h = np.array([-2,0,3,-3,0,0])


solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input,output = solver.solve(c)

histplot(output)

plot3d(output, A=[0,1,2], B=[3,4,5])

vin_vout(input, output, p_bit=5)