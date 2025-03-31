import numpy as np

from qiskit.quantum_info import Operator


def state_fidelity_fitness(o1: Operator, o2: Operator) -> float:
    """
    Calculate the state fidelity between two quantum operators

    Args:
        o1: Generated circuits Qiskit operator
        o2: Target circuit Qiskit operator

    Returns:
        Float value representing the state fidelity between the two operators
    """

    if o1 is None or o2 is None:
        return 0.0

    if o1.dim != o2.dim:
        return 0.0

    try:
        channel = o1.compose(o2.adjoint())
        input_dim, _ = channel.dim
        fid = np.abs(np.trace(channel.data) / input_dim) ** 2
        return np.real(fid)
    except Exception as e:
        print(f"Error in state fidelity fitness calculation: {e}")
        return 0.0
