from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout
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
def AND_gate():
    J = np.array([[0,-1,2], [-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h


def four_bit_Adder():
    J = np.array([[0, 0, 0, 3, -3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 2, 2, 0, 1, -2, 0, -2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-3, 2, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, -3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 3, 2, 0, 1, -2, 0, -2, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, -3, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -3, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 2, 0, 1, -2, 0, -2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, -3, 2, 1, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -3, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 2, 0, 1, -2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, -3, 2, 1, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 2, 0]])
    h = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return J, h


# NO. of nodes or p-bits:
num_nodes = 64
circuit = CircuitBuilder(num_nodes)



# Define nodes to the gates

AND_gate_J, AND_gate_h = AND_gate()
circuit.add_gate(AND_gate_J, AND_gate_h, [55,59,63])
circuit.add_gate(AND_gate_J, AND_gate_h, [56,59,1])
circuit.add_gate(AND_gate_J, AND_gate_h, [57,59,7])
circuit.add_gate(AND_gate_J, AND_gate_h, [58,59,12])

circuit.add_gate(AND_gate_J, AND_gate_h, [55,60,0])
circuit.add_gate(AND_gate_J, AND_gate_h, [56,60,6])
circuit.add_gate(AND_gate_J, AND_gate_h, [57,60,11])
circuit.add_gate(AND_gate_J, AND_gate_h, [58,60,16])


circuit.add_gate(AND_gate_J, AND_gate_h, [55,61,21])
circuit.add_gate(AND_gate_J, AND_gate_h, [56,61,22])
circuit.add_gate(AND_gate_J, AND_gate_h, [57,61,23])
circuit.add_gate(AND_gate_J, AND_gate_h, [58,61,24])


circuit.add_gate(AND_gate_J, AND_gate_h, [55,62,38])
circuit.add_gate(AND_gate_J, AND_gate_h, [56,62,39])
circuit.add_gate(AND_gate_J, AND_gate_h, [57,62,40])
circuit.add_gate(AND_gate_J, AND_gate_h, [58,62,41])


# adder
four_bit_Adder_J, four_bit_Adder_h = four_bit_Adder()
circuit.add_gate(four_bit_Adder_J, four_bit_Adder_h, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])


circuit.add_gate(four_bit_Adder_J, four_bit_Adder_h, [21,10,25,26,27,33,22,15,28,29,34,23,20,30,31,35,24,18,37,32,36])


circuit.add_gate(four_bit_Adder_J, four_bit_Adder_h, [38,34,42,43,44,50,39,35,45,46,51,40,36,47,48,52,41,37,54,49,53])



# Combined circuit matrix and bias vector
J_final, h_final = circuit.get_circuit()



c = PCircuit(num_nodes)


c.J = J_final
c.h = h_final


print(J_final)
print(h_final)


#solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.9)

#input,output = solver.solve(c)


#histplot(output)


current_dir = os.getcwd()
print("Current Directory:", current_dir)

# File paths
j_file_path = os.path.join(current_dir, 'J_final.txt')
h_file_path = os.path.join(current_dir, 'h_final.txt')

# Save J_final (2D array) as integers
np.savetxt(j_file_path, J_final, fmt='%d', delimiter=',')
print(f"J_final data saved to {j_file_path}")

# Save h_final (1D array) as integers
np.savetxt(h_file_path, h_final, fmt='%d', delimiter=',')
print(f"h_final data saved to {h_file_path}")