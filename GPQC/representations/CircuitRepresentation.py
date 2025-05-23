from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
from qiskit import QuantumCircuit


class CircuitRepresentation(ABC):
    def __init__(
            self,
            num_qubits: int,
            max_depth: int,
            gate_set: Dict[str, Any],
            gate_probs: np.array,
            random_gate_prob: float = 0.7,
    ):
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.gate_set = gate_set
        self.gate_probs = gate_probs
        self.random_gate_prob = random_gate_prob

        self.qiskit_circuit = None
        self.qiskit_operator = None

    @abstractmethod
    def generate_random_circuit(self) -> None:
        """
        Generate a random quantum circuit
        """

        pass

    @abstractmethod
    def mutate(
            self,
            mutation_rate: float,
            mutation_type: str,
            generation: int,
            max_generations: int,
            max_mutations: int = 1,
    ) -> None:
        """
        Apply mutation operations to the circuit

        Args:
            mutation_rate: Probability of applying mutation
            mutation_type: Type of mutation operation
            generation: Current generation number
            max_generations: Maximum number of generations
            max_mutations: Maximum number of mutations to apply
        """

        pass

    @abstractmethod
    def crossover(
            self, other: "CircuitRepresentation", crossover_rate: float = 0.7
    ) -> "CircuitRepresentation":
        """
        Create a new circuit by combining this circuit with another

        Args:
            other: Another circuit representation to crossover with
            crossover_rate: Probability of performing crossover

        Returns:
            A new circuit representation resulting from the crossover
        """

        pass

    @abstractmethod
    def replicate(self) -> "CircuitRepresentation":
        """
        Create a deep copy of this circuit representation

        Returns:
            A new identical copy of this circuit representation
        """

        pass

    @abstractmethod
    def calculate_similarity(self, other: "CircuitRepresentation") -> float:
        """
        Calculate similarity between this circuit and another circuit

        Args:
            other: Another CircuitRepresentation object to compare with

        Returns:
            Float value between 0.0 and 1.0, where 1.0 means identical circuits
            and 0.0 means completely different circuits.
        """

        pass

    @abstractmethod
    def _to_qiskit(self) -> QuantumCircuit:
        pass
