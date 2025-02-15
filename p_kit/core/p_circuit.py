# import numpy as np
#
#
# class PCircuit:
#
#     """Create and holds J and h parameters.
#
#     Parameters
#     ----------
#     n_pbits: string
#         Identifier of the pipeline (for log purposes).
#
#     Attributes
#     ----------
#     h : np.array((n_pbits, 1))
#         biases
#     J : np.array((n_pbits, n_pbits))
#         weights
#
#     Notes
#     -----
#      versionadded:: 0.0.1
#
#     """
#
#     def __init__(self, n_pbits):
#         self.n_pbits = n_pbits
#         self.h = np.zeros((n_pbits, 1))
#         self.J = np.zeros((n_pbits, n_pbits))
#
#     def set_weight(self, from_pbit, to_pbit, weight, sym=True):
#         self.J[from_pbit, to_pbit] = weight
#         if sym:
#             self.J[to_pbit, from_pbit] = weight
#
#


import numpy as np
from numba import cuda


class PCircuit:
    def __init__(self, n_pbits):
        self.n_pbits = n_pbits
        self.h = cuda.to_device(np.zeros((n_pbits, 1), dtype=np.float32))
        self.J = cuda.to_device(np.zeros((n_pbits, n_pbits), dtype=np.float32))

    @staticmethod
    @cuda.jit
    def _set_weight_kernel(J, from_pbit, to_pbit, weight, sym):
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bdx = cuda.blockDim.x
        idx = bx * bdx + tx

        if idx == 0:  # Only the first thread executes this
            J[from_pbit, to_pbit] = weight
            if sym:
                J[to_pbit, from_pbit] = weight

    def set_weight(self, from_pbit, to_pbit, weight, sym=True):
        threadsperblock = 32
        blockspergrid = 1
        self._set_weight_kernel[blockspergrid, threadsperblock](self.J, from_pbit, to_pbit, weight, sym)
        cuda.synchronize()

    def get_weights(self):
        return self.J.copy_to_host(), self.h.copy_to_host()
