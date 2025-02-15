# from .p_circuit import PCircuit
#
# __all__ = ["PCircuit"]



from .p_circuit import PCircuit

__all__ = ["PCircuit"]

# Ensure that necessary CUDA libraries are initialized if needed
try:
    from numba import cuda
    device = cuda.get_current_device()
    device.reset()  # Reset device to avoid potential conflicts
except Exception as e:
    print(f"Warning: CUDA initialization failed - {e}")
