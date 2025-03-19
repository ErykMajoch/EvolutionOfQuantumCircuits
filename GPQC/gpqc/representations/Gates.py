from qiskit.circuit.library import *

SUPPORTED_GATES = {
    # Single qubit gates
    "X": {
        "num_qubits": 1,
        "category": "single",
    },
    "Y": {
        "num_qubits": 1,
        "category": "single",
    },
    "Z": {
        "num_qubits": 1,
        "category": "single",
    },
    "H": {
        "num_qubits": 1,
        "category": "single",
    },
    "S": {
        "num_qubits": 1,
        "category": "single",
    },
    "SX": {
        "num_qubits": 1,
        "category": "single",
    },
    # Rotation gates
    "Rx": {
        "num_qubits": 1,
        "category": "rotation",
    },
    "Ry": {
        "num_qubits": 1,
        "category": "rotation",
    },
    "Rz": {
        "num_qubits": 1,
        "category": "rotation",
    },
    # Phase gates
    "R": {
        "num_qubits": 1,
        "category": "phase",
    },
    # Two qubit gates
    "SWAP": {
        "num_qubits": 2,
        "category": "swap",
    },
    "CX": {
        "num_qubits": 2,
        "category": "controlled",
    },
    "CY": {
        "num_qubits": 2,
        "category": "controlled",
    },
    "CZ": {
        "num_qubits": 2,
        "category": "controlled",
    },
    # Three qubit gates
    "CCX": {
        "num_qubits": 3,
        "category": "controlled",
    },
}

QISKIT_GATES = {
    "X": XGate(),
    "Y": YGate(),
    "Z": ZGate(),
    "H": HGate(),
    "S": SGate(),
    "SX": SXGate(),
    "Rx": RXGate,
    "Ry": RYGate,
    "Rz": RZGate,
    "R": PhaseGate,
    "SWAP": SwapGate(),
    "CX": CXGate(),
    "CY": CYGate(),
    "CZ": CZGate(),
    "CCX": CCXGate(),
}
