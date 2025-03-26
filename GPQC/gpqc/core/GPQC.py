import random
from typing import Dict, Optional, Tuple, Any, Type

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

from gpqc.algorithms.selection import tournament_selection
from gpqc.evaluation.evaluation_functions import matrix_similarity
from gpqc.representations.Base import CircuitRepresentation
from gpqc.representations.Gates import SUPPORTED_GATES
from gpqc.representations.Tree.QTree import QTree


class GPQC:
    """
    Genetic Programming for Quantum Circuits (GPQC) core class

    Attributes:
        population_size: Number of individuals in the population
        max_generations: Maximum number of generations to evolve
        fitness_threshold: Target fitness value to stop evolution
        crossover_rate: Probability of applying crossover operation
        mutation_rate: Probability of applying mutation operation
        diversity_threshold: Threshold below which diversity preservation kicks in
        selection_method: Method used for parent selection
        tournament_size: Size of tournaments when using tournament selection
        num_qubits: Number of qubits in each circuit
        max_depth: Maximum depth of each circuit
        gate_set: Dictionary of available gates and their properties
        gate_probs: Array of normalized probabilities for gate selection
        target_matrix: Target unitary matrix to approximate
        representation_class: Circuit representation class to use
        population: Array of circuit individuals
        best_individual: Best circuit found during evolution
        best_fitness: Highest fitness value achieved
        fitness_scores: Array of fitness scores for the current population
        best_fitness_history: List tracking best fitness per generation
        avg_fitness_history: List tracking average fitness per generation
        current_generation: Current generation number
        adaptive_rate: Adaptive rate for crossover and mutation operations
        stagnation_counter: Counter to track how many consecutive generations have experienced stagnation
        recently_stagnated: Boolean flag to indicate if the algorithm recently restarted due to stagnation
    """

    def __init__(
        self,
        requirements: Dict[str, Any],
        repr_class: Optional[Type[CircuitRepresentation]] = QTree,
    ):
        """
        Initialise the GPQC evolutionary algorithm class

        Args:
            requirements: Dictionary containing GP and circuit parameters
            repr_class: Circuit representation class to use (default: QTree)
        """
        gp_params = requirements.get("gp_behaviour", {})
        circuit_params = requirements.get("circuit_behaviour", {})

        # Evolution parameters
        self.population_size = gp_params.get("population_size", 100)

        termination_params = gp_params.get("termination_params", {})
        self.max_generations = termination_params.get("max_generations", 50)
        self.fitness_threshold = termination_params.get("fitness_threshold", 0.95)
        self.crossover_rate = gp_params.get("crossover_rate", 0.7)
        self.mutation_rate = gp_params.get("mutation_rate", 0.3)
        self.diversity_threshold = gp_params.get("diversity_threshold", 0.3)

        selection_params = gp_params.get("selection_params", {})
        self.selection_method = selection_params.get("selection_method", "tournament")
        if self.selection_method == "tournament":
            self.tournament_size = selection_params.get("tournament_size", 3)

        # Circuit parameters
        self.num_qubits = circuit_params.get("qubits", 3)
        self.max_depth = circuit_params.get("max_depth", 6)
        self.gate_set, self.gate_probs = self._prepare_gate_set(
            circuit_params.get("gates", {})
        )
        self.target_matrix = circuit_params.get("target_matrix", None)

        # Class parameters
        self.representation_class = repr_class
        self.population = np.array([], dtype=self.representation_class)
        self.best_individual = None
        self.best_fitness = 0.0
        self.fitness_scores = np.array([], dtype=float)
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.current_generation = 0
        self.adaptive_rate = 0.0
        self.stagnation_counter = 0
        self.recently_stagnated = False

        self.crossover_rate_history = []
        self.mutation_rate_history = []
        self.max_mutations_history = []

    def initialise_population(self, population_size: int) -> None:
        """
        Initialise the population with random quantum circuits of given representation

        Args:
            population_size: The population size to initialise
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

    def evaluate_population(self) -> None:
        """
        Evaluate fitness of all individuals in the population.
        """
        self.fitness_scores = np.ndarray((self.population_size,), dtype=float)
        for index, individual in enumerate(self.population):
            unitary = individual.unitary_matrix
            self.fitness_scores[index] = matrix_similarity(unitary, self.target_matrix)

    def calculate_diversity(self) -> float:
        """
        Calculate diversity measure of the population

        Returns:
            Float value representing population diversity (0-1)
        """
        diversity_sum = 0.0
        total_comparisons = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                similarity = self.population[i].calculate_similarity(self.population[j])
                diversity_sum += 1.0 - similarity
                total_comparisons += 1

        return diversity_sum / total_comparisons if total_comparisons > 0 else 0

    def apply_fitness_sharing(self) -> np.ndarray:
        """
        Apply fitness sharing to promote diversity

        Returns:
            Array of shared fitness values
        """
        shared_fitness = self.fitness_scores.copy()
        sharing_radius = 0.4

        for i in range(len(self.population)):
            niche_count = 1.0

            for j in range(len(self.population)):
                if i != j:
                    similarity = self.population[i].calculate_similarity(
                        self.population[j]
                    )

                    if similarity > sharing_radius:
                        niche_count += (similarity - sharing_radius) / (
                            1 - sharing_radius
                        )

            shared_fitness[i] = self.fitness_scores[i] / max(1.0, niche_count)

        return shared_fitness

    def select_parents(self) -> np.array:
        """
        Select parent circuits for breeding the next generation

        Returns:
            Array of selected parent circuits
        """
        # If diversity is low, apply fitness sharing
        diversity = self.calculate_diversity()
        fitness_for_selection = self.fitness_scores
        if diversity < self.diversity_threshold:
            fitness_for_selection = self.apply_fitness_sharing()

        match self.selection_method:
            case "tournament":
                return tournament_selection(
                    population=self.population,
                    fitness_scores=fitness_for_selection,
                    generation=self.current_generation,
                    max_generations=self.max_generations,
                    base_tournament_size=self.tournament_size,
                )
            case _:
                return random.choices(
                    self.population,
                    weights=fitness_for_selection,
                    k=self.population_size,
                )

    def evolve(self, plot_results: bool = False) -> None:
        """
        Run the evolutionary algorithm to find a circuit approximating the target

        Args:
            plot_results: Whether to plot fitness progress (default: False)
        """
        self.initialise_population(self.population_size)

        self.best_individual = None
        self.best_fitness = 0.0
        self.fitness_scores = np.array([], dtype=float)
        self.current_generation = 0
        self.adaptive_rate = 0.0
        self.stagnation_counter = 0
        self.recently_stagnated = False

        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.crossover_rate_history = []
        self.mutation_rate_history = []
        self.max_mutations_history = []

        for generation in range(self.max_generations):
            self.current_generation = generation

            if self.current_generation == 0 or self.recently_stagnated:
                self.evaluate_population()
                self.recently_stagnated = False

            if (
                self.current_generation > 50
                and abs(self.best_fitness - self.best_fitness_history[-50]) < 0.001
            ):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter > 30:
                print(
                    f"Restarting at generation {self.current_generation + 1} due to stagnation"
                )

                elite_count = max(5, int(0.1 * self.population_size))
                elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
                elite = np.array(
                    [self.population[i].replicate() for i in elite_indices]
                )

                self.initialise_population(self.population_size - elite_count)
                self.population = np.append(self.population, elite)

                self.recently_stagnated = True
                self.stagnation_counter = 0

                continue

            parents = self.select_parents()

            # Calculate adaptive rates
            self._calculate_adaptive_rate()

            crossover_rate = self.crossover_rate
            if self.current_generation < self.max_generations * 0.3:
                crossover_rate *= 1.2
            else:
                crossover_rate *= 0.8

            mutation_rate = self.mutation_rate
            if self.current_generation > self.max_generations * 0.7:
                mutation_rate *= 1.5
            crossover_rate *= self.adaptive_rate
            mutation_rate *= self.adaptive_rate

            max_mutations = max(
                1, int(self.num_qubits * self.max_depth * self.adaptive_rate)
            )

            # Create new population with adaptive elitism
            elite_count = max(
                3,
                int(
                    self.population_size
                    * 0.1
                    * (1 - self.current_generation / self.max_generations)
                ),
            )
            elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
            new_population = [self.population[i].replicate() for i in elite_indices]

            while len(new_population) < self.population_size:

                parent_indices = np.random.choice(len(parents), 2, replace=False)
                parent1, parent2 = (
                    parents[parent_indices[0]],
                    parents[parent_indices[1]],
                )

                child = parent1.crossover(parent2, crossover_rate)
                mutation_type = random.choices(
                    ["replace", "parameter", "insert", "delete"],
                    weights=[0.5, 0.3, 0.1, 0.1],
                )[0]

                child.mutate(
                    mutation_rate,
                    mutation_type,
                    self.current_generation,
                    self.max_generations,
                    max_mutations,
                )
                new_population.append(child)

            self.population = np.array(new_population)
            self.evaluate_population()

            # Track progress
            best_index = np.argmax(self.fitness_scores)
            best_fitness = self.fitness_scores[best_index]

            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(self.fitness_scores))

            self.crossover_rate_history.append(crossover_rate)
            self.mutation_rate_history.append(mutation_rate)
            self.max_mutations_history.append(max_mutations)

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_individual = self.population[best_index].replicate()

            # Check termination condition
            if self.best_fitness >= self.fitness_threshold:
                print(
                    f"Reached fitness threshold at generation {self.current_generation}"
                )
                break

            # Log process
            if generation % 10 == 0:
                diversity = self.calculate_diversity()
                print(
                    f"Generation {self.current_generation}: Best fitness = {self.best_fitness:.4f},"
                    f" Average fitness = {np.mean(self.fitness_scores):.4f},"
                    f" Diversity = {diversity:.2f}"
                )

        print(f"Evolution completed after {self.current_generation + 1} generations")
        print(f"Best fitness achieved: {self.best_fitness:.4f}")

        if plot_results:
            self._plot_results()

    def get_best_circuit(self) -> Optional[QuantumCircuit]:
        """
        Get the best quantum circuit found during evolution

        Returns:
            Qiskit QuantumCircuit of the best individual, or None if none found
        """
        if self.best_individual:
            return self.best_individual.qiskit_circuit
        return None

    def _prepare_gate_set(self, gate_set: Dict) -> Tuple[Dict, np.array]:
        """
        Prepare the set of quantum gates available for circuit generation

        Args:
            gate_set: Dictionary mapping gate names to their selection weights

        Returns:
            Tuple containing filtered gate dictionary and normalised probabilities
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

    def _calculate_adaptive_rate(self) -> None:
        """
        Calculate adaptive rate for crossover and mutation operations
        """
        linear_component = max(
            0.5, 1 - (self.current_generation / self.max_generations)
        )
        exponential_component = np.exp(
            -self.current_generation / (self.max_generations / 4)
        )

        self.adaptive_rate = 0.7 * linear_component + 0.3 * exponential_component

    def _plot_results(self) -> None:
        """
        Plot the evolution of fitness across generations
        """
        plt.figure(figsize=(12, 12))

        # Plot fitness
        plt.subplot(3, 1, 1)
        plt.plot(self.best_fitness_history, "b-", label="Best Fitness")
        plt.plot(self.avg_fitness_history, "r-", label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Evolution")
        plt.legend()
        plt.grid(True)

        # Plot fitness difference (to show exploration)
        plt.subplot(3, 1, 2)
        fitness_gap = [
            best - avg
            for best, avg in zip(self.best_fitness_history, self.avg_fitness_history)
        ]
        plt.plot(fitness_gap, "g-", label="Fitness Gap (Exploration Indicator)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Gap")
        plt.title("Exploration Indicator (Best Fitness - Average Fitness)")
        plt.grid(True)
        plt.legend()

        # Plot rates
        plt.subplot(3, 1, 3)
        plt.plot(self.crossover_rate_history, "b-", label="Crossover Rate")
        plt.plot(self.mutation_rate_history, "r-", label="Mutation Rate")

        # Plot max mutations on secondary y-axis for better visibility (as it's an integer)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(self.max_mutations_history, "g--", label="Max Mutations")
        ax2.set_ylabel("Max Mutations")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.xlabel("Generation")
        ax1.set_ylabel("Rate")
        plt.title("Parameter Evolution During Search")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("fitness_history.png")
        plt.close()
