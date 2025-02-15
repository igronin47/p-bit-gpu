"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout, plot3d, energyplot
import numpy as np
'''import cupy as cp'''
import matplotlib.pyplot as plt
import os

c = PCircuit(7)

c.J = np.array([[ 0,  0,  1,  2,  1, -2,  0],
 [ 0,  0, -1,  0,  2,  0,  0],
 [ 1, -1,  0, -2,  2,  0,  0],
 [ 2,  0, -2,  0,  0, -1,  2],
 [ 1,  2,  2,  0,  0,  2,  0],
 [-2,  0,  0, -1,  2,  0,  2],
 [ 0,  0,  0,  2,  0,  2,  0]
])
c.h = np.array([ 0, -1,  0,  3,  1,  3, 10])



solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

energy, output = solver.solve(c)

histplot(output)

