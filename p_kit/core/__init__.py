
from .p_circuit import PCircuit

__all__ = ["PCircuit"]

# Ensure CUDA is available before initializing
try:
    from numba import cuda
    if cuda.is_available():
        device = cuda.get_current_device()
        print(f"CUDA initialized on device: {device.name}")
    else:
        print("Warning: CUDA is not available. Running on CPU.")
except Exception as e:
    print(f"Warning: Failed to initialize CUDA - {e}")
