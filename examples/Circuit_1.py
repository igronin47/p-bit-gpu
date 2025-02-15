
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
