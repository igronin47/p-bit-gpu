import numpy as np

class SATCircuit:
    def __init__(self, num_vars, total_nodes):
        """
        Initialize the circuit with the given number of variables and total nodes.
        total_nodes includes intermediate outputs.
        """
        self.num_vars = num_vars
        self.total_nodes = total_nodes
        self.J = np.zeros((total_nodes, total_nodes))
        self.h = np.zeros(total_nodes)
        self.current_node = num_vars  # Start intermediate nodes after variables

    def add_2_input_gate(self, input1, input2, gate_matrix, gate_bias, negations):
        """
        Adds a 2-input gate between two inputs.
        negations: A tuple (neg1, neg2) where neg1/neg2 is True if input1/input2 is negated.
        Returns the new output node created for this gate.
        """
        new_node = self.current_node
        self.current_node += 1

        # Expand the J and h matrices to include the new node
        size = self.current_node
        if size > self.J.shape[0]:
            self.J = np.pad(self.J, ((0, 1), (0, 1)), mode='constant')
            self.h = np.pad(self.h, (0, 1), mode='constant')

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

def parse_cnf_and_build():
    """
    Hardcoded CNF example:
        1  -3 0
        2   3  -1 0
    """
    clauses = [[1, -3], [2, 3, -1]]
    num_vars = 3  # Variables: 1, 2, 3
    total_nodes = 7  # Total nodes including intermediates

    circuit = SATCircuit(num_vars, total_nodes)

    # Clause 1: 1 OR -3
    inputs_clause1 = [0, 2]  # Node indices for literals 1 and -3
    negations_clause1 = [False, True]
    or_output1 = circuit.add_2_input_gate(inputs_clause1[0], inputs_clause1[1], *OR_gate(), negations_clause1)

    # Clause 2: 2 OR 3 OR -1
    inputs_clause2 = [1, 2, 0]  # Node indices for literals 2, 3, and -1
    negations_clause2 = [False, False, True]
    # Chain OR gates for 3-input OR
    intermediate_or = circuit.add_2_input_gate(inputs_clause2[0], inputs_clause2[1], *OR_gate(), negations_clause2[:2])
    or_output2 = circuit.add_2_input_gate(intermediate_or, inputs_clause2[2], *OR_gate(), (False, negations_clause2[2]))

    # Combine with AND gate
    and_output = circuit.add_2_input_gate(or_output1, or_output2, *AND_gate(), (False, False))

    circuit.finalize()
    return circuit.get_circuit()


# Run the circuit builder
J, h = parse_cnf_and_build()

print("Final Coupling Matrix J:")
print(J)
print("\nFinal Bias Vector h:")
print(h)
