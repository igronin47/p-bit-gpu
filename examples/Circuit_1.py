# from p_kit.core import PCircuit
# from p_kit.solver.csd_solver import CaSuDaSolver
# from p_kit.visualization import histplot, vin_vout
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# #Circuit-AND Gate followed by COPY Gate
#
# c = PCircuit(8)
#
# #c.J = np.array([[0,-1,-1,1],[-1,0,1,-1],[-1,1,0,1],[1,-1,1,0]])
# #c.h = np.array([0,0,0,0])
#
# #c.J = np.array([[0,-2,-1,-1,2,2,2,2],[-2,0,2,2,2,2,2,2],[-1,2,0,-2,2,2,2,2],[-1,2,-2,0,2,2,2,2],[2,2,2,2,0,-2,-2,-2],[2,2,2,2,-2,0,-2,-2],[2,2,2,2,-2,-2,0,-2],[2,2,2,2,-2,-2,-2,0]])
# #c.h = np.array([-1,1,1,-1,0,0,0,0])
#
# #c.J = np.array([[0,-1,0,2,0],[-1,0,0,2,0],[0,0,0,-1,2],[2,2,-1,0,2],[0,0,2,2,0]])
# #c.h = np.array([1,1,1,-1,-2])
#
#
# c.J = np.array([
#  [ 0,-1, 2, 0, 0,  0, 0, 0],
#  [-1,  0,  2, -1,  2,  0,  0,  0],
#  [ 2,  2,  0,  0,  0,  3, -3,  2],
#  [ 0, -1,  0,  0,  2,  0,  0,  0],
#  [ 0,  2,  0,  2,  0,  2,  2,  0],
#  [ 0,  0,  3,  0,  2,  0,  1, -2],
#  [ 0,  0, -3,  0,  2,  1,  0,  2],
#  [ 0,  0,  2,  0,  0, -2,  2,  0]])
#
# c.h = np.array([1,  2, -2,  1,  0, -2, -2,  0])
#
#
#
# solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.5)
#
# input, output = solver.solve(c)
#
# histplot(output)
#
# #vin_vout(input, output, p_bit=3)
#
#
# from numba import cuda
# print("CUDA Available:", cuda.is_available())
# print("CUDA Device Name:", cuda.get_current_device().name)


import os
os.environ["NUMBA_CUDA_PTX_VERSION"] = "8.5"  # Force PTX version 8.5
os.environ["NUMBA_CUDA_ARCH"] = "6.1"  # Your GTX 1050 has Compute Capability 6.1

from numba import cuda
print("CUDA Available:", cuda.is_available())
print("CUDA Device Name:", cuda.get_current_device().name)



from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < x.size:
        out[idx] = x[idx] + y[idx]

# Initialize data
N = 10
h_x = np.arange(N, dtype=np.float32)
h_y = np.arange(N, dtype=np.float32)
h_out = np.zeros(N, dtype=np.float32)

# Allocate memory on GPU
d_x = cuda.to_device(h_x)
d_y = cuda.to_device(h_y)
d_out = cuda.to_device(h_out)

# Launch kernel
threads_per_block = 32
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)

# Copy result back
h_out = d_out.copy_to_host()
print("Result:", h_out)
