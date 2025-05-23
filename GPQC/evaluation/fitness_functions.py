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


def circuit_depth_fitness(depth: int, max_depth: int) -> float:
    """
    Calculate the fitness based on circuit depth

    Args:
        depth: Depth of the circuit
        max_depth: Maximum allowed depth

    Returns:
        Float value representing the normalised circuit depth (lower is better)
    """

    if depth is None or max_depth is None:
        return 0.0

    if max_depth <= 0:
        return 0.0

    try:
        return depth / max_depth
    except Exception as e:
        print(f"Error in circuit depth fitness calculation: {e}")
        return 0.0


def gate_count_fitness(count: int, max_gates: int) -> float:
    """
    Calculate the fitness based on gate count

    Args:
        count: Number of gates in the circuit
        max_gates: Maximum allowed number of gates

    Returns:
        Float value representing the normalised gate count (lower is better)
    """

    if count is None or max_gates is None:
        return 0.0

    if max_gates <= 0:
        return 0.0

    try:
        return count / max_gates
    except Exception as e:
        print(f"Error in gate count fitness calculation: {e}")
        return 0.0
