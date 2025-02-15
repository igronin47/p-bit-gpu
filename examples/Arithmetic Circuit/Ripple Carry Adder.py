"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout
import numpy as np
import matplotlib.pyplot as plt
import os


c = PCircuit(15)


c.J = np.array([[0,-2,-1,-1,2,2,0,0,0,0,0,0,0,0,0],[-2,0,-1,-1,2,2,0,0,0,0,0,0,0,0,0],[-1,-1,0,-1,-1,2,-1,0,0,0,0,0,0,2,0],[-1,-1,-1,0,2,0,0,0,0,0,0,0,0,0,0],[2,2,-1,2,0,0,-1,0,0,0,0,0,0,2,0],[2,2,2,0,0,0,-1,-1,-1,-1,-1,2,-1,0,2],[0,0,-1,0,-1,-1,0,0,0,0,0,0,0,2,0],[0,0,0,0,0,-1,0,0,-1,-1,2,2,0,0,0],[0,0,0,0,0,-1,0,-1,0,-1,2,2,0,0,0],[0,0,0,0,0,-1,0,-1,-1,0,2,0,0,0,0],[0,0,0,0,0,-1,0,2,2,2,0,0,-1,0,2],[0,0,0,0,0,2,0,2,2,0,0,0,-1,0,0],[0,0,0,0,0,-1,0,0,0,0,-1,-1,0,0,2],[0,0,2,0,2,0,2,0,0,0,0,0,0,0,0],[0,0,0,0,0,2,0,0,0,0,2,0,2,0,0]])
c.h = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)

#histplot(output)

#np.set_printoptions(threshold=np.inf)
#print(output)


#print(output)

#vin_vout(input, output, p_bit=4)


# Get the current working directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'array_output1.txt')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")

