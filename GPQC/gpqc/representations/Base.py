from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from qiskit import QuantumCircuit

from gpqc.representations.Gates import SUPPORTED_GATES


class CircuitRepresentation(ABC):
    """
    Abstract base class for quantum circuit representations

    Attributes:
        num_qubits: Number of qubits in the circuit
        max_depth: Maximum depth of the circuit
        gate_set: Dictionary of available gates and their properties
        gate_probs: Array of normalised probabilities for gate selection
        random_gate_prob: Probability of adding a gate at each position
    """

    @abstractmethod
    def __init__(
            self,
            num_qubits: int,
            max_depth: int,
            gate_set: Dict[str, float],
            random_gate_prob: float = 0.7,
    ):
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.gate_set, self.gate_probs = self._prepare_gate_set(gate_set)
        self.random_gate_prob = random_gate_prob

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
        Create a deep copy of this circuit representation.

        Returns:
            A new identical copy of this circuit representation
        """
        pass

    @abstractmethod
    def to_qiskit(self) -> QuantumCircuit:
        """
        Convert this circuit representation to a Qiskit QuantumCircuit object

        Returns:
            Qiskit QuantumCircuit representing this circuit
        """
        pass

    def _prepare_gate_set(self, gate_set: dict) -> tuple[dict, np.array]:
        """
        Prepare the set of available gates and their selection probabilities

        Args:
            gate_set: Dictionary mapping gate names to their selection weights

        Returns:
            Tuple containing the filtered gate set and normalised probabilities
        """
        available_gates = {
            gate: SUPPORTED_GATES[gate]
            for gate in gate_set
            if gate in SUPPORTED_GATES
               and SUPPORTED_GATES[gate]["num_qubits"] <= self.num_qubits
        }

        if available_gates == {}:
            raise ValueError("No valid gates found")

        probabilities = np.array([gate_set[gate] for gate in available_gates.keys()])
        normalised_probs = probabilities / probabilities.sum()
        return available_gates, normalised_probs
