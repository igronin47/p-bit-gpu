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




def adder():
    J = np.array(
        [[0, -2, 0, 2, 2, 0], [-2, 0, 0, 2, 2, 0], [0, 0, 0, 3, -3, 2], [2, 2, 3, 0, 1, -2], [2, 2, -3, 1, 0, 2],
         [0, 0, 2, -2, 2, 0]])
    h = np.array([0, 0, 0, 0, 0, 0])
    return J, h


def half_adder():
      J = np.array([[0, 0, 3, -3, 2], [0, 0, 2, 2, 0], [3, 2, 0, 1, -2], [-3, 2, 1, 0, 2], [2, 0, -2, 2, 0]])

      h = np.array([0, 2, -2, -2, 0])
      return J, h

def XOR_gate():
    J = np.array([[0,-2,2,2,0],[-2,0,2,2,0],[2,2,0,1,2],[2,2,1,0,-2],[0,0,2,-2,0]])

    h = np.array([0,0,3,-3,-2])
    return J, h


def AND_gate():
    J = np.array([[0,-1,2], [-1,0,2],[2,2,0]])
    h = np.array([1,1,-2])
    return J, h




num_nodes = 62
circuit = CircuitBuilder(num_nodes)



#FULL ADDER
adder_J, adder_h = adder()
circuit.add_gate(adder_J, adder_h, [5,6,3,7,8,9])
circuit.add_gate(adder_J, adder_h, [10,11,8,12,13,14])
circuit.add_gate(adder_J, adder_h, [9,19,17,20,21,22])
circuit.add_gate(adder_J, adder_h, [14,23,21,24,25,26])
circuit.add_gate(adder_J, adder_h, [22,34,32,35,36,37])
circuit.add_gate(adder_J, adder_h, [26,38,36,39,40,41])
circuit.add_gate(adder_J, adder_h, [29,42,40,43,44,45])



#HALF ADDER
half_adder_J, half_adder_h = half_adder()
circuit.add_gate(half_adder_J, half_adder_h, [0,1,2,3,4])
circuit.add_gate(half_adder_J, half_adder_h, [4,15,16,17,18])
circuit.add_gate(half_adder_J, half_adder_h, [13,25,27,28,29])
circuit.add_gate(half_adder_J, half_adder_h, [18,30,31,32,33])
circuit.add_gate(half_adder_J, half_adder_h, [28,44,46,47,48])


#AND GATE
and_gate_J, and_gate_h = AND_gate()
circuit.add_gate(and_gate_J, and_gate_h, [50,49,1])
circuit.add_gate(and_gate_J, and_gate_h, [49,51,6])
circuit.add_gate(and_gate_J, and_gate_h, [49,52,11])

circuit.add_gate(and_gate_J, and_gate_h, [54,53,15])
circuit.add_gate(and_gate_J, and_gate_h, [54,55,19])
circuit.add_gate(and_gate_J, and_gate_h, [54,56,23])

circuit.add_gate(and_gate_J, and_gate_h, [58,57,30])
circuit.add_gate(and_gate_J, and_gate_h, [58,59,34])
circuit.add_gate(and_gate_J, and_gate_h, [58,60,38])
circuit.add_gate(and_gate_J, and_gate_h, [58,61,42])


# Combined circuit matrix and bias vector
J_final, h_final = circuit.get_circuit()



c = PCircuit(num_nodes)


c.J = J_final
c.h = h_final


print(J_final)
print(h_final)




current_dir = os.getcwd()
print("Current Directory:", current_dir)

# File paths
j_file_path = os.path.join(current_dir, 'J_final_PP1.txt')
h_file_path = os.path.join(current_dir, 'h_final_PP1.txt')

# Save J_final (2D array) as integers
np.savetxt(j_file_path, J_final, fmt='%d', delimiter=',')
print(f"J_final data saved to {j_file_path}")

# Save h_final (1D array) as integers
np.savetxt(h_file_path, h_final, fmt='%d', delimiter=',')
print(f"h_final data saved to {h_file_path}")
