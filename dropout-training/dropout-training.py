import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x_arr = np.array(x)

    if rng is not None:
        random_vals = rng.random(x_arr.shape)
    else:
        random_vals = np.random.random(x_arr.shape)

    scale = 1.0 / (1.0 - p)
    dropout_pattern = np.where(random_vals > p, scale, 0)

    output = x_arr * dropout_pattern
    return (output, dropout_pattern)