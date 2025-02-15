# # from p_kit.core.p_circuit import PCircuit
# # from p_kit.solver.csd_solver import CaSuDaSolver
# # from p_kit.visualization import histplot
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import networkx as nx
# #
# # # Define the city graph (distance matrix)
# # city_graph = np.array([
# #     [0, 10, 15],
# #     [10, 0, 20],
# #     [15, 20, 0]
# # ])
# #
# # Nm = len(city_graph)  # Number of cities
# #
# # # Construct J matrix
# # J = np.zeros((Nm ** 2, Nm ** 2))
# #
# # # Rule 3: Negative distances between cities
# # for i in range(Nm):
# #     for j in range(Nm):
# #         if i != j:
# #             J[j * Nm: j * Nm + Nm, i * Nm: i * Nm + Nm] = -city_graph[j, i]
# #
# # # Rule 1: +1 between p-bits of the same city
# # for i in range(Nm):
# #     J[i * Nm:i * Nm + Nm, i * Nm:i * Nm + Nm] = 1
# #
# # # Rule 2: +1 between p-bits of the same order
# # for i in range(Nm ** 2):
# #     for j in range(Nm ** 2):
# #         if i % Nm == j % Nm:
# #             J[i, j] = 1
# #
# # # Rule 4: 0 on the diagonal
# # np.fill_diagonal(J, 0)
# #
# # # Define bias vector h
# # h = np.zeros(Nm ** 2)
# #
# # # Set start and end constraints (Example: Start at city 0, end at city 2)
# # h[0] = -10  # Force city 0 at position 0
# # h[2 * Nm + 2] = 10  # Force city 2 at position Nm-1
# #
# # # Initialize p-circuit
# # circuit = PCircuit(Nm ** 2)
# # circuit.J = J
# # circuit.h = h
# #
# # solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)
# # input_signals, output_states = solver.solve(circuit)
# #
# #
# #
# # # Debugging: Print a few sample outputs
# # print("\nSample Output States (First 10):\n", output_states[:10])
# #
# #
# # # Extract the best path
# # def extract_tsp_path(output_states, Nm):
# #     """Extracts the most frequent TSP path from sampled states."""
# #     path_counts = {}
# #     for state in output_states:
# #         path = []
# #         for order in range(Nm):
# #             for city in range(Nm):
# #                 if state[city * Nm + order] == 1:
# #                     path.append(city)
# #                     break
# #         path_tuple = tuple(path)
# #         path_counts[path_tuple] = path_counts.get(path_tuple, 0) + 1
# #
# #     return max(path_counts, key=path_counts.get, default=None)
# #
# #
# # best_path = extract_tsp_path(output_states, Nm)
# # print("Best Path Found:", best_path)
# #
# #
# # # Visualize the TSP solution
# # def visualize_tsp_solution(best_path, city_graph):
# #     """ Visualizes the best TSP path using NetworkX. """
# #     G = nx.DiGraph()
# #     pos = {i: (np.cos(2 * np.pi * i / Nm), np.sin(2 * np.pi * i / Nm)) for i in range(Nm)}
# #     for i in range(Nm):
# #         G.add_node(i, pos=pos[i])
# #     for i in range(len(best_path) - 1):
# #         G.add_edge(best_path[i], best_path[i + 1], weight=city_graph[best_path[i]][best_path[i + 1]])
# #     G.add_edge(best_path[-1], best_path[0], weight=city_graph[best_path[-1]][best_path[0]])
# #
# #     plt.figure(figsize=(6, 6))
# #     labels = nx.get_edge_attributes(G, 'weight')
# #     nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='r', node_size=1000, font_size=15, arrows=True)
# #     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# #     plt.title("Most Common TSP Solution Path")
# #     plt.show()
# #
# #
# # if best_path:
# #     visualize_tsp_solution(best_path, city_graph)
# #
# # # Plot histogram
# # histplot(output_states)
# #
# # print(J)
# # print(h)
#
#
# from p_kit.core.p_circuit import PCircuit
# from p_kit.solver.csd_solver import CaSuDaSolver
# from p_kit.visualization import histplot
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
#
# # Define the city graph (distance matrix)
# city_graph = np.array([
#     [0, 100, 15,500],
#     [100, 0, 20,40],
#     [15, 20, 0,30],
#     [500,40,30,0]
# ])
#
# Nm = len(city_graph)  # Number of cities
#
# # Construct J matrix
# J = np.zeros((Nm ** 2, Nm ** 2))
#
# # Rule 3: Negative distances between cities
# for i in range(Nm):
#     for j in range(Nm):
#         if i != j:
#             J[j * Nm: j * Nm + Nm, i * Nm:i * Nm + Nm] = -city_graph[j, i]
#
# # Rule 1: +1 between p-bits of the same city
# for i in range(Nm):
#     J[i * Nm:i * Nm + Nm, i * Nm:i * Nm + Nm] = 1
#
# # Rule 2: +1 between p-bits of the same order
# for i in range(Nm ** 2):
#     for j in range(Nm ** 2):
#         if i % Nm == j % Nm:
#             J[i, j] = 1
#
# # Rule 4: 0 on the diagonal
# np.fill_diagonal(J, 0)
#
# # Define bias vector h
# h = np.zeros(Nm ** 2)
# #
# # # Set start and end constraints (Example: Start at city 0, end at city 2)
# # h[0] = -10  # Force city 0 at position 0
# # h[2 * Nm + 2] = 10  # Force city 2 at position Nm-1
#
# # Initialize p-circuit
# circuit = PCircuit(Nm ** 2)
# circuit.J = J
# circuit.h = h
#
# solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.3)
# input_signals, output_states = solver.solve(circuit)
#
# # 🔥 Heatmap Visualization
# plt.figure(figsize=(6, 6))
# plt.imshow(output_states[-1, :].reshape((Nm, Nm)), cmap='hot', interpolation='nearest')
# plt.colorbar(label="Activation (0 or 1)")
# plt.title("Final TSP Solution Heatmap")
# plt.xlabel("Tour Step")
# plt.ylabel("Cities")
# plt.show()
#
#
# # Extract the best path
# def extract_tsp_path(sample_matrix):
#     """ Extracts the TSP path from a binary matrix. """
#     path = []
#     for order in range(sample_matrix.shape[1]):  # Iterate through tour steps
#         city = np.argmax(sample_matrix[:, order])  # Find city with value '1'
#         path.append(city)
#     return path
#
#
# final_sample = output_states[-1, :].reshape((Nm, Nm))
# best_path = extract_tsp_path(final_sample)
# print("Best TSP Path Found:", best_path)
#
#
# # 📌 Graph-Based Visualization
# def visualize_tsp_route(path, city_graph):
#     """ Visualizes the best TSP path using NetworkX. """
#     G = nx.DiGraph()
#
#     # Define city positions in a circular layout
#     pos = {i: (np.cos(2 * np.pi * i / Nm), np.sin(2 * np.pi * i / Nm)) for i in range(Nm)}
#
#     # Add nodes (cities)
#     for i in range(Nm):
#         G.add_node(i, pos=pos[i])
#
#     # Add edges based on best path
#     for i in range(len(path) - 1):
#         G.add_edge(path[i], path[i + 1], weight=-city_graph[path[i]][path[i + 1]])
#
#     # Add edge to complete the tour (last city → first city)
#     G.add_edge(path[-1], path[0], weight=-city_graph[path[-1]][path[0]])
#
#     # Draw the graph
#     plt.figure(figsize=(6, 6))
#     labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='r', node_size=1000, font_size=15, arrows=True)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#
#     plt.title("TSP Solution Path")
#     plt.show()
#
#
# # 🏆 Visualize the Best Path Found
# visualize_tsp_route(best_path, city_graph)


from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define the city graph (distance matrix)
city_graph = np.array([
    [0, 100, 15, 500],
    [100, 0, 20, 40],
    [15, 20, 0, 30],
    [500, 40, 30, 0]
])

Nm = len(city_graph)  # Number of cities

# Construct J matrix
J = np.zeros((Nm ** 2, Nm ** 2))

# Rule 3: Negative distances between cities
for i in range(Nm):
    for j in range(Nm):
        if i != j:
            J[j * Nm: j * Nm + Nm, i * Nm:i * Nm + Nm] = -city_graph[j, i]

# Rule 1: +1 between p-bits of the same city
for i in range(Nm):
    J[i * Nm:i * Nm + Nm, i * Nm:i * Nm + Nm] = 10

# Rule 2: +1 between p-bits of the same order
for i in range(Nm ** 2):
    for j in range(Nm ** 2):
        if i % Nm == j % Nm:
            J[i, j] = 10

# Rule 4: 0 on the diagonal
np.fill_diagonal(J, 0)

# Define bias vector h
h = np.zeros(Nm ** 2)

# Initialize p-circuit
circuit = PCircuit(Nm ** 2)
circuit.J = J
circuit.h = h

# Solver setup
solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.3)
input_signals, output_states = solver.solve(circuit)

# Heatmap Visualization
plt.figure(figsize=(6, 6))
plt.imshow(output_states[-1, :].reshape((Nm, Nm)), cmap='hot', interpolation='nearest')
plt.colorbar(label="Activation (0 or 1)")
plt.title("Final TSP Solution Heatmap")
plt.xlabel("Tour Step")
plt.ylabel("Cities")
plt.show()


# Extract the best path
def extract_tsp_path(sample_matrix):
    """ Extracts the TSP path from a binary matrix. """
    path = []
    for order in range(sample_matrix.shape[1]):  # Iterate through tour steps
        city = np.argmax(sample_matrix[:, order])  # Find city with value '1'
        path.append(city)
    return path


final_sample = output_states[-1, :].reshape((Nm, Nm))
best_path = extract_tsp_path(final_sample)
print("Best TSP Path Found:", best_path)


# Graph-Based Visualization
def visualize_tsp_route(path, city_graph):
    """ Visualizes the best TSP path using NetworkX. """
    G = nx.DiGraph()

    # Define city positions in a circular layout
    pos = {i: (np.cos(2 * np.pi * i / Nm), np.sin(2 * np.pi * i / Nm)) for i in range(Nm)}

    # Add nodes (cities)
    for i in range(Nm):
        G.add_node(i, pos=pos[i])

    # Add edges based on best path
    for i in range(len(path) - 1):
        G.add_edge(path[i], path[i + 1], weight=-city_graph[path[i]][path[i + 1]])

    # Add edge to complete the tour (last city → first city)
    G.add_edge(path[-1], path[0], weight=-city_graph[path[-1]][path[0]])

    # Draw the graph
    plt.figure(figsize=(6, 6))
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='r', node_size=1000, font_size=15, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("TSP Solution Path")
    plt.show()


# Visualize the Best Path Found
visualize_tsp_route(best_path, city_graph)

# Histogram Visualization
histplot(output_states)
