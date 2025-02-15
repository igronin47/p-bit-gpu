"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt
import os




c = PCircuit(4)


#c.J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])

#Bias of Majority Gate
#c.h = np.array([1,1,-2])

#Bias of AND Gate using Majority Gate
#c.h = np.array([-2,0,0,0])

#Bias of OR Gate using Majority Gate
#c.h = np.array([2,0,0,0])


#Bias of 2:1 MUX using Majority Gate

#c.J = np.array([[0,-1,0,-1,2,0,0,0,0],[-1,0,-1,-1,2,-1,2,0,0],[0,-1,0,-1,0,-1,2,0,0],[-1,-1,-1,0,2,0,0,0,0],[2,2,0,2,0,0,-1,-1,2],[0,-1,-1,0,0,0,2,0,0],[0,2,2,0,-1,2,0,-1,2],[0,0,0,0,-1,0,-1,0,2],[0,0,0,0,2,0,2,2,0]])
#c.h = np.array([-2,-2,2,0,0,2,0,2,0])


#Full Adder using 3 Majority Gate
#c.J = np.array([[0,-2,-1,2,0,-1,2,0],[-2,0,-1,2,0,-1,2,0],[-1,-1,0,2,-1,-1,-1,2],[2,2,2,0,-1,0,0,0],[0,0,-1,-1,0,0,-1,2],[-1,-1,-1,0,0,0,2,0],[2,2,-1,0,-1,2,0,2],[0,0,2,0,2,0,2,0]])
#c.h = np.array([-3,3,3,0,0,0,0,0])


#Half Adder using 3 Majority Gate
#c.J = np.array([[0,-2,-1,2,0,-1,2,0],[-2,0,-1,2,0,-1,2,0],[-1,-1,0,2,-1,-1,-1,2],[2,2,2,0,-1,0,0,0],[0,0,-1,-1,0,0,-1,2],[-1,-1,-1,0,0,0,2,0],[2,2,-1,0,-1,2,0,2],[0,0,2,0,2,0,2,0]])
#c.h = np.array([0,0,-2,0,0,0,0,0])


#Half Adder using 3 Majority Gate
#c.J = np.array([[0,-2,-1,2,0,-1,2,0],[-2,0,-1,2,0,-1,2,0],[-1,-1,0,2,0,-1,0,0],[2,2,2,0,-2,0,-1,2],[0,0,0,-2,0,0,-1,2],[-1,-1,-1,0,0,0,2,0],[2,2,0,-1,-1,2,0,2],[0,0,0,2,2,0,2,0]])
#c.h = np.array([0,0,-2,0,0,0,0,0])



#Majority gate + inverter input
#c.J = np.array([[0,-2,2,2],[-2,0,2,2],[2,2,0,-2],[2,2,-2,0]])
#c.h = np.array([0,0,0,0])


#c.J = np.array([[0,1,1,1],[1,0,1,-2],[1,1,0,-2],[1,-2,-2,0]])
#c.h = np.array([0,0,0,0])

#c.J = np.array([[0,1,1,-2],[1,0,1,-2],[1,1,0,-2],[-2,-2,-2,0]])
#c.h = np.array([0,0,0,0])

#c.J = np.array([[0,-1,-1,1],[-1,0,-1,1],[-1,-1,0,1],[1,1,1,0]])

#c.J = np.array([[0,-1,1,2],[-1,0,1,2],[1,1,0,-2],[2,2,-2,0]])

"""
c.J = np.array([[0,-1,-1,2],
                [-1,0,-1,2],
                [-1,-1,0,2],
                [2,2,2,0]])
"""


c.J = np.array([[0,-1,-1,-1,2],
                [-1,0,-1,-1,2],
                [-1,-1,0,-1,2],
                [-1,-1,-1,0,2],
                [2,2,2,2,0]])


c.h = np.array([0,0,0,0,0])


solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.3)

input, output = solver.solve(c)

histplot(output)

#vin_vout(input, output, p_bit=3)

#np.set_printoptions(threshold=np.inf)
#print(output)


"""
# Get the current working directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'output_3_input_AND.csv')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")
"""





#3d Histogram plot for the p-bit
#plot3d(output, A=[0,1,2,3], B=[4,5,6,7])


"""
vin_vout(input, output, p_bit=3)

"""