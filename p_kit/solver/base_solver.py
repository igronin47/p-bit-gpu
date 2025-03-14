class Solver:
    def __init__(self, Nt, dt, i0, expected_mean=0) -> None:
        self.Nt = Nt  # Number of iterations (time steps)
        self.dt = dt  # Time step size
        self.i0 = i0  # Input scaling factor
        self.expected_mean = expected_mean  # Future use (e.g., adjusting bias dynamically)

    def solve(self):
        raise NotImplementedError("Subclasses must implement the solve method")
