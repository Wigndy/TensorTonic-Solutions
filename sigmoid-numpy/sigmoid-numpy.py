import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    a = np.array(x, dtype=float)
    res =  1 / (1 + np.exp(-a))
    return float(a) if isinstance(x, float) else res