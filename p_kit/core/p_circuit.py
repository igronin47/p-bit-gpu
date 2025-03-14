import numpy as np
from numba import cuda


class PCircuit:
    def __init__(self, n_pbits):
        self.n_pbits = n_pbits
        self.h = np.zeros((n_pbits, 1), dtype=np.float32)  # Store as NumPy first
        self.J = np.zeros((n_pbits, n_pbits), dtype=np.float32)  # Store as NumPy first

    @staticmethod
    @cuda.jit
    def _set_weight_kernel(J_flat, n_pbits, from_pbit, to_pbit, weight, sym):
        idx = cuda.grid(1)
        if idx == 0:  # Ensure only one thread modifies the matrix
            J_flat[from_pbit * n_pbits + to_pbit] = weight
            if sym:
                J_flat[to_pbit * n_pbits + from_pbit] = weight

    def set_weight(self, from_pbit, to_pbit, weight, sym=True):
        n_pbits = self.n_pbits

        # Transfer data to GPU
        d_J = cuda.to_device(self.J.flatten())  # Flatten to 1D for better memory access

        # Configure CUDA kernel
        threadsperblock = 32
        blockspergrid = 1  # Only need 1 block since 1 element is modified

        # Launch kernel
        self._set_weight_kernel[blockspergrid, threadsperblock](d_J, n_pbits, from_pbit, to_pbit, weight, sym)
        cuda.synchronize()

        # Copy back to CPU and reshape
        self.J = d_J.copy_to_host().reshape((n_pbits, n_pbits))

    def get_weights(self):
        return self.J, self.h  # No need for extra `.copy_to_host()`, since they are CPU-side
