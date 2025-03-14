# from p_kit.core import PCircuit
# from p_kit.solver.csd_solver import CaSuDaSolver
# from p_kit.visualization import histplot, vin_vout
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# #Circuit-AND Gate followed by COPY Gate
#
# c = PCircuit(9)
# #
# # c.J = np.array([[0,1,-2,0],
# #                 [1,0,2,0],
# #                 [-2,2,0,0],
# #                 [0,0,0,0]
# #                 ])
# # c.h = np.array([-3,3,-1,-3])
#
#
#
# #c.J = np.array([[0,-2,-1,-1,2,2,2,2],[-2,0,2,2,2,2,2,2],[-1,2,0,-2,2,2,2,2],[-1,2,-2,0,2,2,2,2],[2,2,2,2,0,-2,-2,-2],[2,2,2,2,-2,0,-2,-2],[2,2,2,2,-2,-2,0,-2],[2,2,2,2,-2,-2,-2,0]])
# #c.h = np.array([-1,1,1,-1,0,0,0,0])
#
#
# #
# # c.J = np.array([[0,-2,1,2,0],
# #                 [-2,0,1,0,2],
# #                 [1,1,0,-2,-2],
# #                 [2,0,-2,0,0],
# #                 [0,2,-2,0,0]
# #                 ])
# # c.h = np.array([3,3,3,0,0])
# #
# #
#
# #
# # c.J = np.array([[0,-2,1,2,0,0],
# #                  [-2,0,1,0,2,0],
# #                  [1,1,0,-2,-2,0],
# #                 [2,0,-2,0,0,0],
# #                 [0,2,-2,0,0,0],
# #                 [0,0,0,0,0,0]
# #                  ])
# #
# # c.h = np.array([5,5,5,-1,-1,-4])
#
#
# c.J = np.array([[0,-2,-2,-2,1,2,0,0,0],
#                 [-2,0,-2,-2,1,0,2,0,0],
#                 [-2,-2,0,-2,1,0,0,2,0],
#                 [-2,-2,-2,0,1,0,0,0,2],
#                 [1,1,1,1,0,-2,-2,-2,-2],
#                 [2,0,0,0,-2,0,0,0,0],
#                 [0,2,0,0,-2,0,0,0,0],
#                 [0,0,2,0,-2,0,0,0,0],
#                 [0,2,0,0,-2,0,0,0,0],
#                  ])
#
# c.h = np.array([5,-5,5,5,-5,-1,-1,-1,0])
#
#
#
#
# solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.5)
#
# input, output = solver.solve(c)
#
# histplot(output)
#
# #vin_vout(input, output, p_bit=3)

import sys
import os

# Get the absolute path of the parent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)  # Add it to Python's module search path

# Now import PCircuit
from p_kit.core import PCircuit

import numpy as np
from numba import cuda

# Define GPU kernel
@cuda.jit
def update_pbits_gpu(pbits, weights):
    i = cuda.grid(1)  # Get thread index
    if i < pbits.shape[0]:  # Ensure valid index
        net_input = 0
        for j in range(pbits.shape[0]):
            net_input += weights[i, j] * pbits[j]
        pbits[i] = 1 if net_input > 0 else -1

# ---------- CONFIGURATION ----------
n = 8192  # Increase number of p-bits for better GPU utilization
threads_per_block = 256  # Common choice for performance
blocks_per_grid = max(128, (n + threads_per_block - 1) // threads_per_block)  # Ensure at least 128 blocks

print(f"Using {blocks_per_grid} blocks and {threads_per_block} threads per block")

# ---------- INITIALIZATION ----------
pbits = np.random.choice([-1, 1], size=n).astype(np.int32)
weights = np.random.randn(n, n).astype(np.float32)

# Transfer data to GPU
d_pbits = cuda.to_device(pbits)
d_weights = cuda.to_device(weights)

# ---------- GPU COMPUTATION ----------
update_pbits_gpu[blocks_per_grid, threads_per_block](d_pbits, d_weights)

# Transfer results back to CPU
pbits = d_pbits.copy_to_host()

print("GPU computation completed successfully!")
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda  # Ensure CUDA is properly imported

# Create a 2-bit P-Circuit
c = PCircuit(2)

# Send J and h to GPU before passing to solver
c.J = cuda.to_device(np.array([[0, 1], [1, 0]], dtype=np.float32))
c.h = cuda.to_device(np.array([0, 0], dtype=np.float32))

# Initialize the GPU-accelerated solver
solver = CaSuDaSolver(Nt=200000, dt=0.1667, i0=2)

# Solve using GPU
input, output = solver.solve(c)

# Convert output to CPU before plotting
output = output.copy()  # Ensure it's on CPU for visualization

from numba import cuda
print(f"Running on GPU: {cuda.get_current_device().name}")


from numba import cuda
import time

# Confirm GPU execution
if cuda.is_available():
    device = cuda.get_current_device()
    print(f"Running on GPU: {device.name}")

# Time the GPU execution
start_time = time.time()
input, output = solver.solve(c)
cuda.synchronize()  # Ensure all GPU tasks finish
end_time = time.time()

print(f"GPU Execution Time: {end_time - start_time:.4f} seconds")

# Plot the results
histplot(output)
