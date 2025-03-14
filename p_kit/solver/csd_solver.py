
from p_kit.core.p_circuit import PCircuit
from .base_solver import Solver
import numpy as np
import matplotlib.pyplot as plt
from p_kit.visualization import histplot
from numba import cuda, float32
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

class CaSuDaSolver(Solver):
    def solve(self, c: PCircuit):
        n_pbits = c.n_pbits

        # Allocate memory on GPU for results
        d_all_m = cuda.device_array((self.Nt, n_pbits), dtype=np.float32)
        d_all_I = cuda.device_array((self.Nt, n_pbits), dtype=np.float32)

        # Initialize states
        I = np.zeros(n_pbits, dtype=np.float32)
        s = np.zeros(n_pbits, dtype=np.float32)
        m = np.sign(0.5 - np.random.rand(n_pbits)).astype(np.float32)

        # Move data to GPU
        d_J = cuda.to_device(c.J.astype(np.float32)) if isinstance(c.J, np.ndarray) else c.J
        d_h = cuda.to_device(c.h.astype(np.float32)) if isinstance(c.h, np.ndarray) else c.h
        d_m = cuda.to_device(m)
        d_I = cuda.to_device(I)
        d_s = cuda.to_device(s)

        # Setup GPU execution parameters
        threadsperblock = 1024  # Increase thread count per block
        blockspergrid = (n_pbits + threadsperblock - 1) // threadsperblock  # Ensure full GPU occupancy

        # Create random states for GPU (needed for randomness inside CUDA kernels)
        rng_states = create_xoroshiro128p_states(n_pbits, seed=42)
        d_rng_states = cuda.to_device(rng_states)

        # Simulation loop
        for run in range(self.Nt):
            CaSuDaSolver.compute_inputs[blockspergrid, threadsperblock](d_J, d_h, d_m, d_I, self.i0, n_pbits)
            CaSuDaSolver.apply_s_function[blockspergrid, threadsperblock](d_I, d_m, d_s, self.dt, n_pbits)
            CaSuDaSolver.update_outputs[blockspergrid, threadsperblock](d_m, d_s, d_rng_states, n_pbits)
            cuda.synchronize()

            # Store results directly in GPU arrays
            d_all_m[run] = d_m
            d_all_I[run] = d_I

        # Copy data back only once after the loop
        all_m = d_all_m.copy_to_host()
        all_I = d_all_I.copy_to_host()

        return all_I, all_m

    @cuda.jit
    def compute_inputs(J, h, m, I, i0, n):
        idx = cuda.grid(1)
        if idx < n:
            sum_Jm = 0.0
            for j in range(n):
                sum_Jm += J[idx, j] * m[j]
            I[idx] = i0 * (sum_Jm + h[idx])

    @cuda.jit
    def apply_s_function(I, m, s, dt, n):
        idx = cuda.grid(1)
        if idx < n:
            s[idx] = math.exp(-dt * math.exp(-m[idx] * I[idx]))

    @cuda.jit
    def update_outputs(m, s, rng_states, n):
        idx = cuda.grid(1)
        if idx < n:
            if s[idx] > xoroshiro128p_uniform_float32(rng_states, idx):
                m[idx] = -m[idx]
