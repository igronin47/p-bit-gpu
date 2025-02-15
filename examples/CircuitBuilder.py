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

def AND_gate_NOT():
    J = np.array([[0,1,-2], [1,0,2],[-2,2,0]])
    h = np.array([-1,1,-2])
    return J, h

def OR_gate():
    J = np.array([[0,-1,2], [-1,0,2],[2,2,0]])
    h = np.array([-1,-1,2])
    return J, h


def bit_adder():
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

def adder():
    J = np.array(
        [[0, -2, 0, 2, 2, 0], [-2, 0, 0, 2, 2, 0], [0, 0, 0, 3, -3, 2], [2, 2, 3, 0, 1, -2], [2, 2, -3, 1, 0, 2],
         [0, 0, 2, -2, 2, 0]])
    h = np.array([0, 0, 0, 0, 0, 0])
    return J, h


def half_adder():
      J = np.array([[0, 0, 3, -3, 2],
                    [0, 0, 2, 2, 0],
                    [3, 2, 0, 1, -2],
                    [-3, 2, 1, 0, 2],
                    [2, 0, -2, 2, 0]])

      h = np.array([0, 2, -2, -2, 0])
      return J, h


def Majority():
    J = np.array([[0,-1,-1,2],[-1,0,-1,2],[-1,-1,0,2],[2,2,2,0]])

    h = np.array([0,0,0,0])
    return J, h


def Majority_NOT():
    J = np.array([[0, 1, 1, -2],
                  [1, 0, -1, 2],
                  [1, -1, 0, 2],
                  [-2, 2, 2, 0]])

    h = np.array([0, 0, 0, 0])
    return J, h


# def POINT_2_DHT_SINGLEBIT():
#     J = np.array()




# NO. of nodes or p-bits:
num_nodes = 8
circuit = CircuitBuilder(num_nodes)


half_adder_J, half_adder_h = half_adder()
circuit.add_gate(half_adder_J,half_adder_h,[2,4,5,6,7])


AND_gate_J, AND_gate_h = AND_gate()
circuit.add_gate(AND_gate_J,AND_gate_h,[0,1,2])
circuit.add_gate(AND_gate_J,AND_gate_h,[3,1,4])




# Define nodes to the gates
#OR_J, OR_h = OR_gate()
#circuit.add_gate(OR_J, OR_h, [0,1,2])
#
#AND_gate_NOT_J, AND_gate_NOT_h = AND_gate_NOT()
#circuit.add_gate(AND_gate_NOT_J, AND_gate_NOT_h, [2,3,4])
#
#AND_gate_J, AND_gate_h = AND_gate()
#circuit.add_gate(AND_gate_J, AND_gate_h, [0,1,3])

#
# #4BIT_ADDER
# bit_adder_J, bit_adder_h = bit_adder()
# circuit.add_gate(bit_adder_J, bit_adder_h, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
#
# #FULL ADDER
# adder_J, adder_h = adder()
# circuit.add_gate(adder_J, adder_h, [40,41,42,0,43,26])
# circuit.add_gate(adder_J, adder_h, [36,37,38,6,39,1])
# circuit.add_gate(adder_J, adder_h, [32,33,34,11,35,7])
# circuit.add_gate(adder_J, adder_h, [29,30,50,16,31,12])
#
#
# #HALF ADDER
# half_adder_J, half_adder_h = half_adder()
# circuit.add_gate(half_adder_J, half_adder_h, [21,22,23,24,25])
# circuit.add_gate(half_adder_J, half_adder_h, [26,23,2,27,28])
# circuit.add_gate(half_adder_J, half_adder_h, [44,45,32,46,36])
# circuit.add_gate(half_adder_J, half_adder_h, [47,48,29,49,33])
#
#
# #AND GATE
# and_gate_J, and_gate_h = AND_gate()
# circuit.add_gate(and_gate_J, and_gate_h, [51,55,59])
# circuit.add_gate(and_gate_J, and_gate_h, [51,56,21])
# circuit.add_gate(and_gate_J, and_gate_h, [51,57,40])
# circuit.add_gate(and_gate_J, and_gate_h, [51,58,44])
#
# circuit.add_gate(and_gate_J, and_gate_h, [52,55,22])
# circuit.add_gate(and_gate_J, and_gate_h, [52,56,41])
# circuit.add_gate(and_gate_J, and_gate_h, [52,57,45])
# circuit.add_gate(and_gate_J, and_gate_h, [52,58,47])
#
# circuit.add_gate(and_gate_J, and_gate_h, [53,55,42])
# circuit.add_gate(and_gate_J, and_gate_h, [53,56,37])
# circuit.add_gate(and_gate_J, and_gate_h, [53,57,48])
# circuit.add_gate(and_gate_J, and_gate_h, [53,58,30])
#
# circuit.add_gate(and_gate_J, and_gate_h, [54,55,38])
# circuit.add_gate(and_gate_J, and_gate_h, [54,56,34])
# circuit.add_gate(and_gate_J, and_gate_h, [54,57,50])
# circuit.add_gate(and_gate_J, and_gate_h, [54,58,17])
#



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

input,output = solver.solve(c)


histplot(output)





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