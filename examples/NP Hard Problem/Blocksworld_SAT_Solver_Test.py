from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np
import os





# Load J and h matrices from files
current_dir = os.getcwd()
j_file_path = os.path.join(current_dir, 'J_SAT.txt')
h_file_path = os.path.join(current_dir, 'h_SAT.txt')

# Load J matrix
with open(j_file_path, 'r') as jf:
    J = np.array(eval(jf.read()))  # Safely parse the formatted matrix

# Load h vector
with open(h_file_path, 'r') as hf:
    h = np.array(eval(hf.read()))  # Safely parse the formatted vector

# Initialize the p-circuit
num_nodes = J.shape[0]
c = PCircuit(num_nodes)

c.J = J
c.h = h

# Configure and run the solver
solver = CaSuDaSolver(Nt=500000, dt=0.1667, i0=0.9)
input, output = solver.solve(c)

# Plot the results
#histplot(output)



#OUTPUT ARRAY IN FILE

current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'output_Blcksworld_sat.csv')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")


# Read data from the file
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'output_Blcksworld_sat.csv')

with open(file_path, 'r') as f:
    lines = f.readlines()

# Combine every set of 5 lines
combined_lines = []
for i in range(0, len(lines), 35):
    combined_line = ' '.join(line.strip() for line in lines[i:i+35])
    combined_lines.append(combined_line)

# Write combined lines to a new file
combined_file_path = os.path.join(current_dir, 'comb_output_Blcksworld_sat.csv')
with open(combined_file_path, 'w') as f:
    for line in combined_lines:
        f.write(line + '\n')

print(f"Combined data saved to {combined_file_path}")
