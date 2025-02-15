from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np
import os

class CircuitBuilder:
    def __init__(self, num_nodes):
        # Initialize the coupling matrix J and bias vector h with zeros
        self.num_nodes = num_nodes
        self.J = np.zeros((num_nodes, num_nodes))
        self.h = np.zeros(num_nodes)

    def add_gate(self, gate_matrix, gate_h, node_mapping):
        """
        Adds a gate matrix and its corresponding bias vector `gate_h` to the circuit.
        `gate_matrix`: Matrix (J) representing the gate's interactions.
        `gate_h`: Bias vector for the gate.
        `node_mapping`: A list of nodes that the gate's matrix corresponds to in the larger circuit.
        """
        size = gate_matrix.shape[0]

        if len(node_mapping) != size:
            raise ValueError("Node mapping size must match the gate matrix size.")

        # Add the small matrix to the appropriate nodes in the larger J matrix
        for i in range(size):
            for j in range(size):
                self.J[node_mapping[i], node_mapping[j]] += gate_matrix[i, j]

        # Add the gate's bias vector to the corresponding nodes in h
        for i in range(size):
            self.h[node_mapping[i]] += gate_h[i]

    def get_circuit(self):
        """Returns the overall coupling matrix J and bias vector h."""
        return self.J, self.h


# Define all distinct gates.
def AND_gate_1():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_2():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_3():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_4():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_5():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_6():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_7():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_8():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_9():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_10():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_11():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_12():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_13():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_14():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_15():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h

def AND_gate_16():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h








def AND_gate_NOT_2():
    J = np.array([[0,1,2],[1,0,-2],[2,-2,0]])
    h = np.array([1,-1,-2])
    return J, h

def AND_gate_NOT_1():
    J = np.array([[0,1,-2],[1,0,2],[-2,2,0]])
    h = np.array([-1,1,-2])
    return J, h
def OR_gate():
    J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
    h = np.array([-1,-1,2])
    return J, h

def MAJORITY_gate():
    J = np.array([[0,-1,-1,2],[-1,0,-1,2],[-1,-1,0,2],[2,2,2,0]])
    h = np.array([0,0,0,0])
    return J, h

def MAJORITY_gate_NOT_1():
    J = np.array([[0,1,1,2],[1,0,-1,2],[1,-1,0,2],[-2,2,2,0]])
    h = np.array([0,0,0,0])
    return J, h

def MAJORITY_gate_NOT_3():
    J = np.array([[0,-1,1,2],[-1,0,1,2],[1,1,0,-2],[2,2,-2,0]])
    h = np.array([0,0,0,0])
    return J, h


def adder_4_bit():
    J = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,2,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,0,0,0,2,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,3,0,0,0,0,0,0,2,0],
[0,0,0,0,0,0,0,0,0,1,3,0,0,0,0,0,0,0,0,0,2],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,2,2,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,-2,2,2,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,-2,2,2,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,-2,2,2,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,-2,0,2,2,0,0,0,0,0,0,0,0,0,0],
[0,0,0,-3,0,0,0,2,2,0,1,0,0,0,0,0,0,0,0,0,2],
[0,0,0,3,0,0,-2,2,2,1,0,2,2,0,0,0,0,0,0,0,-2],
[0,0,-3,0,0,0,2,0,0,0,2,0,1,0,0,0,0,0,0,2,0],
[0,0,3,0,0,-2,2,0,0,0,2,1,0,2,2,0,0,0,0,-2,0],
[0,-3,0,0,0,2,0,0,0,0,0,0,2,0,1,0,0,0,2,0,0],
[0,3,0,0,-2,2,0,0,0,0,0,0,2,1,0,2,2,0,-2,0,0],
[-3,0,0,0,2,0,0,0,0,0,0,0,0,0,2,0,1,2,0,0,0],
[3,0,0,0,2,0,0,0,0,0,0,0,0,0,2,1,0,-2,0,0,0],
[2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,-2,0,0,0,0],
[0,2,0,0,0,0,0,0,0,0,0,0,0,2,-2,0,0,0,0,0,0],
[0,0,2,0,0,0,0,0,0,0,0,2,-2,0,0,0,0,0,0,0,0],
[0,0,0,2,0,0,0,0,0,2,-2,0,0,0,0,0,0,0,0,0,0]
])
    h = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    return J, h

# NO. of nodes or p-bits:
num_nodes = 64
circuit = CircuitBuilder(64)



# Define nodes to the gates



AND_gate_J_1, AND_gate_h_1 = AND_gate_1()
circuit.add_gate(AND_gate_J_1, AND_gate_h_1, [0,4,63])

AND_gate_J_2, AND_gate_h_2 = AND_gate_2()
circuit.add_gate(AND_gate_J_2, AND_gate_h_2, [1,4,8])

AND_gate_J_3, AND_gate_h_3 = AND_gate_3()
circuit.add_gate(AND_gate_J_3, AND_gate_h_3, [2,4,9])

AND_gate_J_4, AND_gate_h_4 = AND_gate_4()
circuit.add_gate(AND_gate_J_4, AND_gate_h_4, [3,4,10])

AND_gate_J_5, AND_gate_h_5 = AND_gate_5()
circuit.add_gate(AND_gate_J_5, AND_gate_h_5, [0,5,12])

AND_gate_J_6, AND_gate_h_6 = AND_gate_6()
circuit.add_gate(AND_gate_J_6, AND_gate_h_6, [1,5,13])

AND_gate_J_7, AND_gate_h_7 = AND_gate_7()
circuit.add_gate(AND_gate_J_7, AND_gate_h_7, [2,5,14])

AND_gate_J_8, AND_gate_h_8 = AND_gate_8()
circuit.add_gate(AND_gate_J_8, AND_gate_h_8, [3,5,15])

AND_gate_J_9, AND_gate_h_9 = AND_gate_9()
circuit.add_gate(AND_gate_J_9, AND_gate_h_9, [0,6,16])

AND_gate_J_10, AND_gate_h_10 = AND_gate_10()
circuit.add_gate(AND_gate_J_10, AND_gate_h_10, [1,6,17])

AND_gate_J_11, AND_gate_h_11 = AND_gate_11()
circuit.add_gate(AND_gate_J_11, AND_gate_h_11, [2,6,18])

AND_gate_J_12, AND_gate_h_12 = AND_gate_12()
circuit.add_gate(AND_gate_J_12, AND_gate_h_12, [3,6,19])

AND_gate_J_13, AND_gate_h_13 = AND_gate_13()
circuit.add_gate(AND_gate_J_13, AND_gate_h_13, [0,7,20])

AND_gate_J_14, AND_gate_h_14 = AND_gate_14()
circuit.add_gate(AND_gate_J_14, AND_gate_h_14, [1,7,21])

AND_gate_J_15, AND_gate_h_15 = AND_gate_15()
circuit.add_gate(AND_gate_J_15, AND_gate_h_15, [2,7,22])

AND_gate_J_16, AND_gate_h_16 = AND_gate_16()
circuit.add_gate(AND_gate_J_16, AND_gate_h_16, [3,7,23])


adder_4_bit_J, adder_4_bit_h = adder_4_bit()
circuit.add_gate(adder_4_bit_J, adder_4_bit_h, [15,14,13,12,11,10,9,8,24.25,26,27,28,29,30,31,32,33,34,35,62])

adder_4_bit_J, adder_4_bit_h = adder_4_bit()
circuit.add_gate(adder_4_bit_J, adder_4_bit_h, [19,18,17,1632,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,61])

adder_4_bit_J, adder_4_bit_h = adder_4_bit()
circuit.add_gate(adder_4_bit_J, adder_4_bit_h, [23,22,21,20,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60])
# For repetition one just give node to their specified j and h matrix
#circuit.add_gate(and_J, and_h, [4, 5])



# Combined circuit matrix and bias vector
J_final, h_final = circuit.get_circuit()



c = PCircuit(num_nodes)


c.J = J_final
c.h = h_final


print(J_final)
print(h_final)


solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.5)

output = solver.solve(c)


#histplot(output)


current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Output array to a file in the current directory
file_path = os.path.join(current_dir, 'output_4bit_MAX.csv')

with open(file_path, 'w') as f:
    for element in output:
        f.write(str(element) + '\n')

print(f"Array data saved to {file_path}")



current_dir = os.getcwd()
print("Current Directory:", current_dir)

# File paths
j_file_path = os.path.join(current_dir, 'J_final_output.txt')
h_file_path = os.path.join(current_dir, 'h_final_output.txt')

# Save J_final (2D array) as integers
np.savetxt(j_file_path, J_final, fmt='%d', delimiter=',')
print(f"J_final data saved to {j_file_path}")

# Save h_final (1D array) as integers
np.savetxt(h_file_path, h_final, fmt='%d', delimiter=',')
print(f"h_final data saved to {h_file_path}")