

"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt
import os




c = PCircuit(6)

#c.J = np.array([[0,-1,-1,-1,-1,2,2,-1],[-1,0,-2,-1,0,0,2,2],[-1,-2,0,-1,0,0,2,2],[-1,-1,-1,0,0,0,0,2],[-1,0,0,0,0,2,-1,-1],[2,0,0,0,2,0,0,2],[2,2,2,0,-1,0,0,0],[-1,2,2,2,-1,2,0,0]])
#c.h = np.array([0,0,-3,0,0,0,0,0])



#Half Subtractor
#c.J = np.array([[0,-1,-1,-1,-1,2,2,-1],[-1,0,-2,-1,0,0,2,2],[-1,-2,0,-1,0,0,2,2],[-1,-1,-1,0,0,0,0,2],[-1,0,0,0,0,2,-1,-1],[2,0,0,0,2,0,0,2],[2,2,2,0,-1,0,0,0],[-1,2,2,2,-1,2,0,0]])
#c.h = np.array([0,0,-3,0,0,0,0,0])


#ARITHMETIC CIRCUITS (REDUCED)

c.J = np.array([[0,0,0,3,-3,2],
                [0,0,-2,2,2,0],
                [0,-2,0,2,2,0],
                [3,2,2,0,1,-2],
                [-3,2,2,1,0,2],
                [2,0,0,-2,2,0]])


c.h = np.array([0,0,0,0,3,3])



solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)

histplot(output)



# Get the current working directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'output_MAJORITY_FS_OUT.txt')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")
