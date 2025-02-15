from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np
import os



class SATCircuit:
    def __init__(self, num_vars):
        """
        Initialize the circuit with the given number of variables.
        The total nodes will expand dynamically as intermediate nodes are added.
        """
        self.num_vars = num_vars
        self.current_node = num_vars  # Start intermediate nodes after variables
        self.J = np.zeros((num_vars, num_vars))
        self.h = np.zeros(num_vars)

    def expand_matrices(self):
        """Expand the size of J and h matrices by one to accommodate a new node."""
        self.J = np.pad(self.J, ((0, 1), (0, 1)), mode='constant')
        self.h = np.pad(self.h, (0, 1), mode='constant')

    def add_2_input_gate(self, input1, input2, gate_matrix, gate_bias, negations):
        """
        Adds a 2-input gate between two inputs.
        negations: A tuple (neg1, neg2) where neg1/neg2 is True if input1/input2 is negated.
        Returns the new output node created for this gate.
        """
        new_node = self.current_node
        self.current_node += 1
        self.expand_matrices()

        # Adjust gate_matrix and gate_bias for negations
        gate_matrix = np.copy(gate_matrix)
        gate_bias = np.copy(gate_bias)
        if negations[0]:
            gate_matrix[:, 0] *= -1
            gate_matrix[0, :] *= -1
            gate_bias[0] *= -1
        if negations[1]:
            gate_matrix[:, 1] *= -1
            gate_matrix[1, :] *= -1
            gate_bias[1] *= -1

        # Add the gate to the circuit
        nodes = [input1, input2, new_node]
        for i in range(3):
            for j in range(3):
                self.J[nodes[i], nodes[j]] += gate_matrix[i, j]
            self.h[nodes[i]] += gate_bias[i]

        return new_node

    def add_multi_input_gate(self, inputs, gate_type):
        """
        Constructs a multi-input gate using 2-input gates.
        Returns the final output node of the gate.
        """
        if gate_type == "AND":
            gate_matrix, gate_bias = AND_gate()
        elif gate_type == "OR":
            gate_matrix, gate_bias = OR_gate()
        else:
            raise ValueError("Unsupported gate type")

        while len(inputs) > 1:
            # Combine the first two inputs using a 2-input gate
            input1 = inputs.pop(0)
            input2 = inputs.pop(0)
            output = self.add_2_input_gate(input1, input2, gate_matrix, gate_bias, (False, False))
            inputs.append(output)  # Add the output node back to the list

        return inputs[0]  # Final output node

    def finalize(self):
        """Symmetrize the J matrix."""
        self.J = (self.J + self.J.T) / 2

    def get_circuit(self):
        return self.J, self.h


# Gate definitions
def AND_gate():
    J = np.array([[0, -1, 2], [-1, 0, 2], [2, 2, 0]])
    h = np.array([1, 1, -2])
    return J, h

def OR_gate():
    J = np.array([[0, -1, 2], [-1, 0, 2], [2, 2, 0]])
    h = np.array([-1, -1, 2])
    return J, h

def parse_cnf_and_build(cnf_file):
    """
    Parses a CNF file and constructs the SAT circuit.
    """
    with open(cnf_file, 'r') as f:
        lines = f.readlines()

    clauses = []
    num_vars = 0

    # Parse CNF
    for line in lines:
        line = line.strip()
        if line.startswith('p'):  # Problem line
            _, _, num_vars, _ = line.split()
            num_vars = int(num_vars)
        elif not line.startswith('c') and line:  # Clause line
            literals = list(map(int, line.split()[:-1]))  # Remove trailing 0
            clauses.append(literals)

    # Build the circuit
    circuit = SATCircuit(num_vars)

    # Add OR gates for each clause
    or_gate_outputs = []
    for clause in clauses:
        inputs = []
        negations = []
        for lit in clause:
            var_index = abs(lit) - 1  # Convert to 0-based indexing
            inputs.append(var_index)
            negations.append(lit < 0)  # True if negated

        # Chain OR gates for multi-input clauses
        while len(inputs) > 1:
            input1 = inputs.pop(0)
            input2 = inputs.pop(0)
            neg1 = negations.pop(0)
            neg2 = negations.pop(0)
            output = circuit.add_2_input_gate(input1, input2, *OR_gate(), (neg1, neg2))
            inputs.append(output)
            negations.append(False)  # Intermediate outputs are not negated

        or_gate_outputs.append(inputs[0])

    # Combine OR gate outputs with an AND gate
    final_output = circuit.add_multi_input_gate(or_gate_outputs, "AND")

    circuit.finalize()
    return circuit.get_circuit()


# Example Usage
cnf_file = "anomaly.cnf"  # Replace with the path to your CNF file
J, h = parse_cnf_and_build(cnf_file)

print("Final Coupling Matrix J:")
print(J)
print("\nFinal Bias Vector h:")
print(h)




current_dir = os.getcwd()
print("Current Directory:", current_dir)

# File paths
j_file_path = os.path.join(current_dir, 'J_SAT.txt')
h_file_path = os.path.join(current_dir, 'h_SAT.txt')

# Save J_final (2D array) as integers
np.savetxt(j_file_path, J, fmt='%d', delimiter=',')
print(f"J_final data saved to {j_file_path}")

# Save h_final (1D array) as integers
np.savetxt(h_file_path, h, fmt='%d', delimiter=',')
print(f"h_final data saved to {h_file_path}")


# Ensure all elements are integers
J = J.astype(int)
h = h.astype(int)

# File paths
current_dir = os.getcwd()
j_file_path = os.path.join(current_dir, 'J_SAT.txt')
h_file_path = os.path.join(current_dir, 'h_SAT.txt')

# Format J matrix
j_formatted = "[\n"  # Start the matrix
for row in J:
    j_formatted += " " + str(list(row)).replace(" ", "") + ",\n"  # Format each row
j_formatted += "]"  # End the matrix

# Format h vector
h_formatted = "[" + ", ".join(map(str, h)) + "]"

# Save to files
with open(j_file_path, 'w') as jf:
    jf.write(j_formatted)
print(f"Formatted J matrix saved to {j_file_path}")

with open(h_file_path, 'w') as hf:
    hf.write(h_formatted)
print(f"Formatted h vector saved to {h_file_path}")


