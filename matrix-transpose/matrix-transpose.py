import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    # n = len(A)
    # m = len(A[0])
    A_arr = np.array(A)
    return np.transpose(A_arr)
