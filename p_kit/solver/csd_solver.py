# from p_kit.core.p_circuit import PCircuit
# from .base_solver import Solver
# from random import random
# import numpy as np
# import matplotlib.pyplot as plt
# from p_kit.visualization import histplot
#
# class CaSuDaSolver(Solver):
#     # K. Y. Camsari, B. M. Sutton, and S. Datta, ‘p-bits for probabilistic spin logic’, Applied Physics Reviews, vol. 6, no. 1, p. 011305, Mar. 2019, doi: 10.1063/1.5055860.
#
#     def solve(self, c: PCircuit):
#         # credit: https://www.purdue.edu/p-bit/blog.html
#         n_pbits = c.n_pbits
#         indices = range(n_pbits)
#
#         all_m = [[]] * self.Nt
#         all_I = [[]] * self.Nt
#
#         I = [0] * n_pbits
#         s = [0] * n_pbits
#         m = [np.sign(0.5 - random()) for _ in indices]
#
#
#         for run in range(self.Nt):
#
#             # compute input biases
#             I = [1 * self.i0 * (np.dot(m, c.J[i]) + c.h[i]) for i in indices]
#
#             # apply S(input)
#             s = [np.exp(-1 * self.dt * np.exp(-1 * m[i] * I[i])) for i in indices]
#
#             #threshold = np.arctanh(self.expected_mean)
#             #s = [np.exp(-1 * self.dt * np.exp(-1 * m[i] * (I[i] + threshold))) for i in indices]
#
#             # compute new output
#             m = [m[i] * np.sign(s[i] - random()) for i in indices]
#
#             all_m[run] = [_ for _ in m]
#             all_I[run] = [_ for _ in I]
#
#
#         return np.array(all_I), np.array(all_m)
#


from p_kit.core.p_circuit import PCircuit
from .base_solver import Solver
from random import random
import numpy as np
import matplotlib.pyplot as plt
from p_kit.visualization import histplot
from numba import cuda, float32


class CaSuDaSolver(Solver):
    def solve(self, c: PCircuit):
        n_pbits = c.n_pbits
        indices = np.arange(n_pbits)

        all_m = np.zeros((self.Nt, n_pbits), dtype=np.float32)
        all_I = np.zeros((self.Nt, n_pbits), dtype=np.float32)

        I = np.zeros(n_pbits, dtype=np.float32)
        s = np.zeros(n_pbits, dtype=np.float32)
        m = np.sign(0.5 - np.random.rand(n_pbits)).astype(np.float32)

        d_J = cuda.to_device(c.J.astype(np.float32))
        d_h = cuda.to_device(c.h.astype(np.float32))
        d_m = cuda.to_device(m)
        d_I = cuda.to_device(I)
        d_s = cuda.to_device(s)

        threadsperblock = 256
        blockspergrid = (n_pbits + (threadsperblock - 1)) // threadsperblock

        for run in range(self.Nt):
            self.compute_inputs[blockspergrid, threadsperblock](d_J, d_h, d_m, d_I, self.i0, n_pbits)
            self.apply_s_function[blockspergrid, threadsperblock](d_I, d_m, d_s, self.dt, n_pbits)
            self.update_outputs[blockspergrid, threadsperblock](d_m, d_s, n_pbits)

            cuda.synchronize()

            all_m[run] = d_m.copy_to_host()
            all_I[run] = d_I.copy_to_host()

        return all_I, all_m

    @cuda.jit
    def compute_inputs(J, h, m, I, i0, n):
        idx = cuda.grid(1)
        if idx < n:
            sum_Jm = 0
            for j in range(n):
                sum_Jm += J[idx, j] * m[j]
            I[idx] = i0 * (sum_Jm + h[idx])

    @cuda.jit
    def apply_s_function(I, m, s, dt, n):
        idx = cuda.grid(1)
        if idx < n:
            s[idx] = np.exp(-dt * np.exp(-m[idx] * I[idx]))

    @cuda.jit
    def update_outputs(m, s, n):
        idx = cuda.grid(1)
        if idx < n:
            if s[idx] > random():
                m[idx] = -m[idx]
