import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    # return float(sum(map(lambda a: a[0] * np.log(a[0] / (a[1] + eps)), zip(p, q))))
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    positive = p_arr > 0
    p_pos = p_arr[positive]
    q_pos = q_arr[positive]

    q_clipped = np.clip(q_pos, eps, None)
    
    diverg = np.sum(p_pos * np.log(p_pos/q_clipped))
    return float(diverg)