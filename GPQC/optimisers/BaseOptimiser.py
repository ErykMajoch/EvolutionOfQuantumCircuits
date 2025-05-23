from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any, Type

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from gpqc.representations.CircuitRepresentation import CircuitRepresentation
from gpqc.representations.Gates import SUPPORTED_GATES
from gpqc.representations.Tree.QTree import QTree


class BaseOptimiser(ABC):
    def __init__(
            self,
            requirements: Dict[str, Any],
            representation: Type[CircuitRepresentation] = QTree,
    ):
        """
        Initialise the base optimiser with specified requirements and representation

        Args:
            requirements: Dictionary containing configuration parameters
            representation: Class defining how quantum circuits are represented
        """

        ga_params = requirements.get("ga_behaviour", {})
        circuit_params = requirements.get("circuit_behaviour", {})

        # Evolution parameters
        self.population_size = ga_params.get("population_size", 100)
        termination_params = ga_params.get("termination_params", {})
        self.max_generations = termination_params.get("max_generations", 50)
        self.fitness_threshold = termination_params.get(
            "fitness_threshold", 0.95)
        self.crossover_rate = ga_params.get("crossover_rate", 0.7)
        self.mutation_rate = ga_params.get("mutation_rate", 0.3)
        self.mutation_types = ["replace", "parameter", "insert", "delete"]
        self.mutation_weights = [0.4, 0.3, 0.15, 0.15]

        # Circuit parameters
        self.num_qubits = circuit_params.get("qubits", 3)
        self.max_depth = circuit_params.get("max_depth", 6)
        self.gate_set, self.gate_probs = self._prepare_gate_set(
            circuit_params.get("gates", {})
        )
        self.target_matrix = circuit_params.get("target_matrix", np.array([]))
        self.target_operator = Operator(self.target_matrix)

        # Optimiser parameters
        self.representation_class = representation
        self.population = np.array([], dtype=self.representation_class)
        self.best_individual = None
        self.best_fidelity = 0.0
        self.current_generation = 0
        self.metrics_history = defaultdict(list)
        self.stagnation_counter = 0
        self.recently_stagnated = False

    @abstractmethod
    def evaluate_population(self, population: np.array = None) -> Optional[np.ndarray]:
        """
        Evaluate the fitness of individuals in the population

        Args:
            population: Population to evaluate, uses self.population if None

        Returns:
            Array of fitness scores for each individual
        """

        pass

    @abstractmethod
    def select_parents(self) -> np.array:
        """
        Select parents from the population for reproduction

        Returns:
            Array of selected parent individuals
        """

        pass

    @abstractmethod
    def run(self, plot_results: bool = False) -> None:
        """
        Run the optimisation process

        Args:
            plot_results: Whether to generate plots of the optimisation results
        """

        pass

    @abstractmethod
    def _plot_results(self) -> None:
        """
        Generate and display plots of optimisation results.
        """

        pass

    def initialise_population(self, population_size: int) -> None:
        """
        Initialise a population of random quantum circuits

        Args:
            population_size: Number of individuals to create
        """

        self.population = np.array(
            [
                self.representation_class(
                    num_qubits=self.num_qubits,
                    max_depth=self.max_depth,
                    gate_set=self.gate_set,
                    gate_probs=self.gate_probs,
                )
                for _ in range(population_size)
            ]
        )
        for individual in self.population:
            individual.generate_random_circuit()

    def get_best_circuit(self) -> Optional[QuantumCircuit]:
        """
        Retrieve the best quantum circuit found during optimisation

        Returns:
            The best quantum circuit if one was found, otherwise None
        """

        if self.best_individual:
            return self.best_individual.qiskit_circuit
        return None

    def _prepare_gate_set(self, gate_set: Dict) -> Tuple[Dict, np.array]:
        """
        Prepare the gate set and probabilities for circuit generation

        Args:
            gate_set: Dictionary mapping gate names to their probabilities

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

        probabilities = np.array([gate_set[gate]
                                 for gate in available_gates.keys()])
        normalised_probs = probabilities / probabilities.sum()
        return available_gates, normalised_probs

    def _calculate_adaptive_rates(self) -> Dict:
        """
        Calculate adaptive rates for crossover and mutation based on current generation

        Returns:
            Dictionary containing adaptive rates for various genetic operations
        """

        linear_component = max(
            0.5, 1 - (self.current_generation + 1 / self.max_generations)
        )
        exponential_component = np.exp(
            -(4 * self.current_generation + 1) / self.max_generations
        )

        adaptive_rate = (0.7 * linear_component) + \
            (0.3 * exponential_component)

        crossover_rate = self.crossover_rate
        if self.current_generation < self.max_generations * 0.5:
            crossover_rate *= 1.2
        else:
            crossover_rate *= 0.8

        mutation_rate = self.mutation_rate
        if self.current_generation > self.max_generations * 0.8:
            mutation_rate *= 1.5

        return {
            "crossover_rate": crossover_rate * adaptive_rate,
            "mutation_rate": mutation_rate * adaptive_rate,
            "max_mutations": max(
                1, int(self.num_qubits * self.max_depth * adaptive_rate)
            ),
            "elite_count": max(
                3,
                int(
                    self.population_size
                    * 0.05
                    * (1 - (-self.current_generation / self.max_generations))
                ),
            ),
        }

    def _create_offspring(self, parents: np.array, size: int) -> np.array:
        """
        Create offspring individuals by crossover and mutation of parents

        Args:
            parents: Array of parent individuals
            size: Number of offspring to create

        Returns:
            Array of new offspring individuals
        """

        adaptive_rates = self._calculate_adaptive_rates()
        offspring = np.ndarray((size,), dtype=self.representation_class)

        # Pre-generate all parent pairs at once
        parent_indices = np.random.choice(
            np.arange(len(parents)), size=(size, 2))
        parent_pairs = parents[parent_indices]

        # Pre-generate all mutation types at once
        mutation_types = np.random.choice(
            self.mutation_types, size=size, p=self.mutation_weights
        )

        for index in range(size):
            parent1, parent2 = parent_pairs[index]
            child = parent1.crossover(
                parent2, adaptive_rates["crossover_rate"])

            child.mutate(
                adaptive_rates["mutation_rate"],
                mutation_types[index],
                self.current_generation,
                self.max_generations,
                adaptive_rates["max_mutations"],
            )

            offspring[index] = child

        return offspring

    def _reset_optimiser(self) -> None:
        """
        Reset the optimiser to its initial state
        """

        self.population = np.array([], dtype=self.representation_class)
        self.best_individual = None
        self.best_fidelity = 0.0
        self.current_generation = 0
        self.metrics_history = defaultdict(list)
        self.stagnation_counter = 0
        self.recently_stagnated = False
