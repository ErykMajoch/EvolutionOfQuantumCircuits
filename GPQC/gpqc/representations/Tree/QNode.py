from typing import Dict, List


class QNode:
    """
    A node in a quantum circuit tree representing a quantum gate

    Attributes:
        gate_type: Type of quantum gate
        gate_category: Category of the gate (e.g., rotation, phase, controlled)
        targets: List of target qubit(s) the gate acts on
        params: Dictionary of parameters for parameterised gates
    """

    def __init__(
            self,
            gate_type: str,
            gate_category: str,
            targets: List,
            params: Dict = None,
    ):
        """
        Initialise a quantum circuit tree node representing a quantum gate.

        Args:
            gate_type: Type of quantum gate
            gate_category: Category of the gate
            targets: List of target qubit indices the gate acts on
            params: Parameters for parameterised gates
        """
        self.gate_type = gate_type
        self.gate_category = gate_category
        self.targets = targets
        self.params = params or {}

    def __repr__(self) -> str:
        """
        Create a string representation of the QNode

        Returns:
            String representation of the QNode
        """
        return f"{self.__class__.__name__}({self.gate_type},target={self.targets},params={self.params})"
