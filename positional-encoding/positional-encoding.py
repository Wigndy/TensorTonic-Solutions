import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """

    pos = np.arange(seq_len).reshape(-1, 1)

    i = np.arange(0, d_model, 2)
    freq = 1 / np.power(base, i/d_model)

    angle = pos * freq
    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, :pe[:, 1::2].shape[1]])
    
    return pe