from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt
import os


#Using cost function
c = PCircuit(3)

c.J = np.array([[0,-1,-2],[-1,0,-2],[-2,-2,0]])
c.h = np.array([1,1,2])
#solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.18)

#Using Hamiltonian Ising constraint Equation
#c = PCircuit(4)

#c.J = np.array([[0,-1,2,0],[-1,0,2,0],[2,2,0,-1],[0,0,-1,0]])
#c.h = np.array([1,1,-2,0])



solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input, output = solver.solve(c)

histplot(output)

#output.remove(3)





#np.set_printoptions(threshold=np.inf)
#print(output)

#3d Histogram plot for the p-bit
#plot3d(output, A=[0,1], B=[2,3])

#vin_vout(input, output, p_bit=3)



"""
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'output_NAND_INV.txt')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")
"""