import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    g_arr = np.array(g)

    g_square = [i * i for i in g_arr]
    scalar = float(np.sqrt(np.sum(g_square)))

    return g_arr if scalar < max_norm else g_arr * max_norm / scalar