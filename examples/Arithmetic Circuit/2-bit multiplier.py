"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout
import numpy as np
import matplotlib.pyplot as plt
import os


c = PCircuit(18)

c.J = np.array([[0,0,-1,-1,-2,0,0,2,2,0,0,0,0,0,0,0,0,0],[0,0,-1,-1,-2,0,0,0,0,0,0,0,2,0,0,0,0,2],[-1,-1,0,0,-2,0,0,2,0,0,0,0,2,0,0,0,0,0],[-1,-1,0,0,-2,0,0,0,2,0,0,0,0,0,0,0,0,2],[-2,-2,-2,-2,0,0,-1,1,0,-1,2,2,0,-1,2,2,0,2],[0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,2,0],[0,0,0,0,-1,0,0,-1,0,0,0,0,0,0,0,2,0,-1],[2,0,2,0,1,0,-1,0,0,0,0,0,0,0,0,2,0,0],[2,0,0,2,0,0,0,0,0,-1,2,0,-1,-1,2,0,0,0],[0,0,0,0,-1,0,0,0,-1,0,2,0,-1,0,0,0,0,0],[0,0,0,0,2,-1,0,0,2,2,0,-1,0,0,0,0,2,0],[0,0,0,0,2,-1,0,0,0,0,-1,0,2,2,0,0,2,0],[0,2,2,0,0,0,0,0,-1,-1,0,2,0,-1,2,0,0,0],[0,0,0,0,-1,0,0,0,-1,0,0,2,-1,0,0,0,0,0],[0,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,0,0],[0,0,0,0,2,0,2,2,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,2,0,0,0,0,2,2,0,0,0,0,0,0],[0,2,0,2,2,0,-1,0,0,0,0,0,0,0,0,0,0,0]])
c.h = np.array([2,2,2,-2,-3,2,0,0,0,0,0,0,0,0,0,0,0,0])

solver = CaSuDaSolver(Nt=100000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)

#histplot(output)



# Get the current working directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'output_Multiplier2.txt')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")


#vin_vout(input, output, p_bit=4)


#OUTPUT ARRAY COMBINING THE LINE
# Read data from the file
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'output_Multiplier2.txt')

with open(file_path, 'r') as f:
    lines = f.readlines()

# Combine every pair of lines
combined_lines = []
for i in range(0, len(lines), 2):
    if i + 1 < len(lines):
        combined_line = lines[i].strip() + ' ' + lines[i+1].strip()
    else:
        combined_line = lines[i].strip()  # In case of an odd number of lines
    combined_lines.append(combined_line)

# Write combined lines to a new file
combined_file_path = os.path.join(current_dir, 'output_Multiplier_OUT2.txt')
with open(combined_file_path, 'w') as f:
    for line in combined_lines:
        f.write(line + '\n')

print(f"Combined data saved to {combined_file_path}")
