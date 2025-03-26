import numpy as np


def matrix_similarity(m1: np.array, m2: np.array) -> float:
    """
    Calculate similarity between two unitary matrices using the Hilbert-Schmidt operator

    Args:
        m1: First unitary matrix
        m2: Second unitary matrix

    Returns:
        Similarity score between 0 and 1
    """
    try:
        if m1 is None or m2 is None:
            return 0.0

        if m1.shape != m2.shape:
            return 0.0

        dim = m1.shape[0]
        inner_product = np.tensordot(m1.conj(), m2, axes=((0, 1), (0, 1)))
        fidelity = np.abs(inner_product) / dim
        return min(1.0, fidelity)
    except Exception as e:
        print(f"Error in matrix similarity calculation: {e}")
        return 0.0
