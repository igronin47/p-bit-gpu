from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout
import numpy as np
import matplotlib.pyplot as plt




c = PCircuit(3)

#c.J = np.array([[0,-2,2,2,0],[-2,0,2,2,0],[2,2,0,1,2],[2,2,1,0,-2],[0,0,2,-2,0]])
#c.h = np.array([0,0,3,-3,-2])

c.J = np.array([[0,1,2],[1,0,-2],[2,-2,0]])
c.h = np.array([1,-1,3])


solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input, output = solver.solve(c)

histplot(output)

#vin_vout(input, output, p_bit=3)


np.set_printoptions(threshold=np.inf)
print(output)