"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt
import os




c = PCircuit(8)


# 2 point dht single bit input
c.J = np.array([
 [ 0,-1, 2, 0, 0,  0, 0, 0],
 [-1,  0,  2, -1,  2,  0,  0,  0],
 [ 2,  2,  0,  0,  0,  3, -3,  2],
 [ 0, -1,  0,  0,  2,  0,  0,  0],
 [ 0,  2,  0,  2,  0,  2,  2,  0],
 [ 0,  0,  3,  0,  2,  0,  1, -2],
 [ 0,  0, -3,  0,  2,  1,  0,  2],
 [ 0,  0,  2,  0,  0, -2,  2,  0]])

c.h = np.array([1,  2, -2,  1,  0, -2, -2,  0])



solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)

histplot(output)
