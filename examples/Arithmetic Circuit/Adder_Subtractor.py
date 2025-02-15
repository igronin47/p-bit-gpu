"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt
import os


c = PCircuit(10)




# 1 bit comparator using digital logic
#c.J = np.array([[0,-1,-1,2,0,0,-1,0],[-1,0,-3,2,-1,2,-1,2],[-1,-3,0,2,-1,2,-1,2],[2,2,2,0,-1,0,0,0],[0,-1,-1,-1,0,2,0,0],[0,2,2,0,2,0,0,0],[-1,-1,-1,0,0,0,0,0],[0,2,2,0,0,0,2,0]])
#c.h = np.array([0,0,0,0,0,0,0,0])



c.J = np.array([[0,-2,-1,2,0,-1,2,0,-1,0],
                [-2,0,-2,2,0,-1,2,0,-1,2],
                [-1,-2,0,2,-1,-1,-1,2,-1,2],
                [2,2,2,0,-1,0,0,0,0,0],
                [0,0,-1,-1,0,0,-1,2,0,0],
                [-1,-1,-1,0,0,0,2,0,0,0],
                [2,2,-1,0,-1,2,0,2,0,0],
                [0,0,2,0,2,0,2,0,0,0],
                [-1,-1,-1,0,0,0,0,0,0,2],
                [0,2,2,0,0,0,0,0,2,0]])


c.h = np.array([0,0,0,0,0,0,0,0,0,0])


solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)



histplot(output)



# Get the current working directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'output_Adder_Subtractor.txt')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")




#3d Histogram plot for the p-bit
#plot3d(output, A=[0,1,2,3], B=[4,5,6,7])

#vin_vout(input, output, p_bit=3)
