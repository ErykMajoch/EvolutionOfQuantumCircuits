from typing import Optional, List, Dict, Any, Type

import matplotlib.pyplot as plt
import numpy as np

from gpqc.evaluation.fitness_functions import (
    state_fidelity_fitness,
    gate_count_fitness,
    circuit_depth_fitness,
)
from gpqc.optimisers.BaseOptimiser import BaseOptimiser
from gpqc.representations import CircuitRepresentation
from gpqc.representations.Tree.QTree import QTree


class NSGA2Optimiser(BaseOptimiser):
    def __init__(
        self,
        requirements: Dict[str, Any],
        representation: Type[CircuitRepresentation] = QTree,
    ) -> None:
        """
        Initialise the NSGA-II optimiser with the given requirements and representation

        Args:
            requirements: Dictionary containing NSGA-II parameters and target specifications
            representation: The circuit representation class to use (default: QTree)
        """

        super().__init__(requirements, representation)

        # Evolution parameters
        nsga_params = requirements.get("nsga_params", {})
        self.objectives = nsga_params.get("objectives", [])
        if not self.objectives:
            self.objectives = [{"name": "fidelity", "minimise": False}]
        self.num_objectives = len(self.objectives)

        # Optimiser parameters
        self.fitness_matrix = np.array([], dtype=float)
        self.crowding_distances = np.array([], dtype=float)
        self.fronts = []

    def evaluate_population(self, population: np.array = None) -> Optional[np.ndarray]:
        """
        Evaluate the fitness of each individual in the population for all objectives

        Args:
            population: Optional population to evaluate (uses self.population if None)

        Returns:
            Matrix of fitness scores for all individuals across all objectives
        """

        if population is None:
            population = self.population

        fitness_matrix = np.zeros((len(population), self.num_objectives))

        max_gates = self.num_qubits * self.max_depth

        for i, individual in enumerate(population):
            operator = individual.qiskit_operator
            circuit_depth = individual.qiskit_circuit.depth()
            gate_count = individual.qiskit_circuit.size()

            for j, objective in enumerate(self.objectives):
                objective_name = objective["name"]
                match objective_name:
                    case "fidelity":
                        fitness = state_fidelity_fitness(
                            operator, self.target_operator)
                    case "gate_count":
                        fitness = gate_count_fitness(gate_count, max_gates)
                    case "circuit_depth":
                        fitness = circuit_depth_fitness(
                            circuit_depth, self.max_depth)
                    case _:
                        raise ValueError(
                            f"Unknown objective: {objective_name}")

                fitness_matrix[i, j] = - \
                    fitness if objective["minimise"] else fitness

        if population is self.population:
            self.fitness_matrix = fitness_matrix
        else:
            return fitness_matrix

    def non_dominated_sort(self, fitness_matrix: np.ndarray) -> List[np.ndarray]:
        """
        Perform non-dominated sorting on the population

        Args:
            fitness_matrix: Matrix of fitness values for all individuals across all objectives

        Returns:
            List of arrays, where each array contains indices of individuals in a Pareto front
        """

        population_size = fitness_matrix.shape[0]

        domination_counts = np.zeros(population_size, dtype=int)
        dominated_sets = [[] for _ in range(population_size)]
        fronts = [[]]

        for i in range(population_size):
            dominates_mask = np.all(
                fitness_matrix[i] >= fitness_matrix, axis=1
            ) & np.any(fitness_matrix[i] > fitness_matrix, axis=1)
            dominated_by_mask = np.all(
                fitness_matrix >= fitness_matrix[i], axis=1
            ) & np.any(fitness_matrix > fitness_matrix[i], axis=1)

            dominated_indices = np.where(dominates_mask)[0]
            dominated_indices = dominated_indices[dominated_indices != i]
            dominated_sets[i].extend(dominated_indices)
            domination_counts[i] = np.sum(dominated_by_mask)

            if domination_counts[i] == 0:
                fronts[0].append(i)

        front_index = 0
        while fronts[front_index]:
            next_front = []
            for i in fronts[front_index]:
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            front_index += 1
            fronts.append(next_front)

        return [np.array(front) for front in fronts[:-1]]

    def calculate_crowding_distance(
        self, fitness_matrix: np.ndarray, front_indices: np.ndarray
    ) -> np.ndarray:
        """
        Calculate crowding distance for individuals in a Pareto front

        Args:
            fitness_matrix: Matrix of fitness values for all individuals
            front_indices: Indices of individuals in the current front

        Returns:
            Array of crowding distances for individuals in the front
        """

        front_size = len(front_indices)
        if front_size <= 2:
            return np.full(front_size, np.inf)

        distances = np.zeros(front_size)

        for index in range(self.num_objectives):
            values = fitness_matrix[front_indices, index]
            sorted_indices = np.argsort(values)

            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            objective_range = values[sorted_indices[-1]
                                     ] - values[sorted_indices[0]]
            if objective_range > 0:
                for i in range(1, front_size - 1):
                    distances[sorted_indices[i]] += (
                        values[sorted_indices[i + 1]] -
                        values[sorted_indices[i - 1]]
                    ) / objective_range

        return distances

    def calculate_all_crowding_distances(
        self, fitness_matrix: np.ndarray, fronts: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate crowding distances for all individuals across all fronts

        Args:
            fitness_matrix: Matrix of fitness values for all individuals
            fronts: List of arrays containing indices of individuals in each front

        Returns:
            Array of crowding distances for all individuals
        """

        population_size = fitness_matrix.shape[0]
        crowding_distances = np.zeros(population_size)

        for front in fronts:
            front_distances = self.calculate_crowding_distance(
                fitness_matrix, front)
            crowding_distances[front] = front_distances

        return crowding_distances

    def select_parents(self) -> np.array:
        """
        Select parents for breeding based on rank and crowding distance

        Returns:
            Array of selected parent individuals
        """

        parents = np.empty(self.population_size, dtype=object)
        front_length = len(self.fronts)

        for i in range(self.population_size):
            index1, index2 = np.random.choice(
                self.population_size, 2, replace=False)

            rank1 = next(
                (j for j, front in enumerate(self.fronts) if index1 in front),
                front_length,
            )
            rank2 = next(
                (j for j, front in enumerate(self.fronts) if index2 in front),
                front_length,
            )

            if rank1 < rank2:
                selected = index1
            elif rank2 < rank1:
                selected = index2
            else:
                if self.crowding_distances[index1] > self.crowding_distances[index2]:
                    selected = index1
                else:
                    selected = index2

            parents[i] = self.population[selected].replicate()

        return parents

    def run(self, plot_results: bool = False) -> None:
        """
        Run the NSGA-II multi-objective optimisation process

        Args:
            plot_results: Whether to generate plots of the optimisation process
        """

        self._reset_optimiser()
        self.initialise_population(self.population_size)
        self.evaluate_population()

        self.fronts = self.non_dominated_sort(self.fitness_matrix)
        self.crowding_distances = self.calculate_all_crowding_distances(
            self.fitness_matrix, self.fronts
        )

        for generation in range(self.max_generations):
            self.current_generation = generation

            # Handle stagnation
            if self.recently_stagnated:
                self.evaluate_population()
                self.recently_stagnated = False

            if (
                self.current_generation > 50
                and abs(self.best_fidelity - self.metrics_history["best_fidelity"][-50])
                < 0.001
            ):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter > 30:
                print(
                    f"Restarting at generation {self.current_generation + 1} due to stagnation"
                )

                # Keep elite individuals from the first front
                elite_count = max(5, int(0.1 * self.population_size))
                elite = np.array(
                    [
                        self.population[i].replicate()
                        for i in self.fronts[0][:elite_count]
                    ]
                )

                # Reinitialise the rest of the population
                self.initialise_population(self.population_size - len(elite))
                self.population = np.append(self.population, elite)

                # Re-evaluate the new population
                self.evaluate_population()
                self.fronts = self.non_dominated_sort(self.fitness_matrix)
                self.crowding_distances = self.calculate_all_crowding_distances(
                    self.fitness_matrix, self.fronts
                )

                self.stagnation_counter = 0
                continue

            # Create offspring
            parents = self.select_parents()
            offspring = self._create_offspring(parents, self.population_size)
            offspring_fitness = self.evaluate_population(offspring)

            combined_population = np.concatenate((self.population, offspring))
            combined_fitness = np.concatenate(
                (self.fitness_matrix, offspring_fitness))
            combined_fronts = self.non_dominated_sort(combined_fitness)
            combined_crowding = self.calculate_all_crowding_distances(
                combined_fitness, combined_fronts
            )

            new_population = []
            new_fitness = []
            new_population_size = 0
            front_index = 0

            # Add complete fronts as long as they fit
            while (
                new_population_size + len(combined_fronts[front_index])
                <= self.population_size
            ):
                for idx in combined_fronts[front_index]:
                    new_population.append(combined_population[idx])
                    new_fitness.append(combined_fitness[idx])
                new_population_size += len(combined_fronts[front_index])
                front_index += 1
                if front_index >= len(combined_fronts):
                    break

            # Add more individuals by crowding distance and take most diverse
            if new_population_size < self.population_size and front_index < len(
                combined_fronts
            ):
                last_front = combined_fronts[front_index]
                crowd_distances = combined_crowding[last_front]
                sorted_indices = np.argsort(-crowd_distances)

                remaining_slots = self.population_size - new_population_size
                for i in range(remaining_slots):
                    idx = last_front[sorted_indices[i]]
                    new_population.append(combined_population[idx])
                    new_fitness.append(combined_fitness[idx])

            self.population = np.array(new_population)
            self.fitness_matrix = np.array(new_fitness)

            self.fronts = self.non_dominated_sort(self.fitness_matrix)
            self.crowding_distances = self.calculate_all_crowding_distances(
                self.fitness_matrix, self.fronts
            )

            # Update the best individual based on fidelity
            first_front = self.fronts[0]
            best_index_in_front = np.argmax(
                self.fitness_matrix[first_front, 0])
            best_individual_index = first_front[best_index_in_front]
            if len(first_front) > 0:
                current_best_fidelity = self.fitness_matrix[best_individual_index, 0]

                if current_best_fidelity > self.best_fidelity:
                    self.best_fidelity = current_best_fidelity
                    self.best_individual = self.population[
                        best_individual_index
                    ].replicate()

            # Track metrics
            self.metrics_history["best_individual"].append(
                np.abs(self.fitness_matrix[best_individual_index])
            )

            # Track objective metrics
            for i, objective in enumerate(self.objectives):
                objective_name = objective["name"]

                # Best value in this objective (from Pareto front)
                if len(first_front) > 0:
                    best_obj_index = np.argmax(
                        self.fitness_matrix[first_front, i])
                    best_obj_value = self.fitness_matrix[first_front[best_obj_index], i]
                    self.metrics_history[f"best_{objective_name}"].append(
                        best_obj_value
                    )
                else:
                    self.metrics_history[f"best_{objective_name}"].append(0)

                # Average value across population
                average_obj_value = np.mean(self.fitness_matrix[:, i])
                self.metrics_history[f"average_{objective_name}"].append(
                    average_obj_value
                )

            adaptive_rates = self._calculate_adaptive_rates()
            for name, rate in adaptive_rates.items():
                self.metrics_history[name].append(rate)

            # Calculate spread as sum of ranges in each objective
            if len(first_front) > 1:
                front_values = self.fitness_matrix[first_front]
                spread = np.sum(np.ptp(front_values, axis=0))
                self.metrics_history["pareto_front_spread"].append(spread)
            else:
                self.metrics_history["pareto_front_spread"].append(0)

            # Store Pareto front
            if generation % 5 == 0 and len(first_front) > 0:
                front_copy = self.fitness_matrix[first_front].copy()
                self.metrics_history["pareto_front"].append(
                    (generation, front_copy))

            if self.best_fidelity >= self.fitness_threshold:
                print(
                    f"Reached fitness threshold at generation {self.current_generation}"
                )
                break

            if generation % 10 == 0:
                average_fitness = np.mean(self.fitness_matrix[:, 0])
                print(
                    f"Generation {self.current_generation}: "
                    f"Best fitness = {self.best_fidelity:.4f}, "
                    f"Average fitness = {average_fitness:.4f}, "
                    f"Pareto front size = {len(self.fronts[0])}, "
                )

        print(
            f"Evolution completed after {self.current_generation + 1} generations")
        print(f"Best fitness achieved: {self.best_fidelity:.4f}")
        print(f"Final Pareto front size: {len(self.fronts[0])}")

        if plot_results:
            self._plot_results()

    def _plot_results(self) -> None:
        """
        Generate plots showing the multi-objective optimisation progress

        Creates a multi-panel figure showing:
        1. Best individual's objective values over generations
        2. Best and average objective values over generations
        3. 3D visualization of the final Pareto front (for 3 objectives)

        Saves the figure to a file named "nsga2_evolution_history.png"
        """

        fig = plt.figure(figsize=(10, 18))

        # Best individual's objective values history
        ax1 = fig.add_subplot(3, 1, 1)
        generations = range(len(self.metrics_history["best_individual"]))
        best_individuals = np.array(self.metrics_history["best_individual"])
        best_individuals = np.abs(best_individuals)

        for i, objective in enumerate(self.objectives):
            objective_name = objective["name"]
            ax1.plot(
                generations, best_individuals[:, i], label=f"{objective_name}")

        ax1.set_title("Best Individual's Objective Values")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Objective Value")
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True)

        # Best and average objective fitness history
        ax2 = fig.add_subplot(3, 1, 2)

        for i, objective in enumerate(self.objectives):
            objective_name = objective["name"]
            best_values = np.abs(
                self.metrics_history[f"best_{objective_name}"])
            avg_values = np.abs(
                self.metrics_history[f"average_{objective_name}"])

            ax2.plot(
                generations,
                best_values,
                label=f"Best {objective_name}",
                color=f"C{i}",
                linestyle="-",
            )
            ax2.plot(
                generations,
                avg_values,
                label=f"Avg {objective_name}",
                color=f"C{i}",
                linestyle="--",
                alpha=0.7,
            )

        ax2.set_title("Best and Average Fitness Values")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Fitness Value")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)

        # Final 3D Pareto front
        if len(self.fronts) > 0 and len(self.fronts[0]) > 0:
            front_values = np.abs(self.fitness_matrix[self.fronts[0]])

            if self.num_objectives == 3:
                ax3 = fig.add_subplot(3, 1, 3, projection="3d")

                fidelity_idx = next(
                    (
                        i
                        for i, obj in enumerate(self.objectives)
                        if obj["name"] == "fidelity"
                    ),
                    0,
                )
                second_idx, third_idx = [
                    i for i in range(self.num_objectives) if i != fidelity_idx
                ]

                x_idx, y_idx, z_idx = second_idx, third_idx, fidelity_idx
                best_idx = np.argmax(front_values[:, fidelity_idx])
                best_point = front_values[best_idx]

                other_indices = np.arange(len(front_values)) != best_idx
                ax3.scatter(
                    front_values[other_indices, x_idx],
                    front_values[other_indices, y_idx],
                    front_values[other_indices, z_idx],
                    c="blue",
                    label="Pareto front",
                )

                ax3.scatter(
                    front_values[best_idx, x_idx],
                    front_values[best_idx, y_idx],
                    front_values[best_idx, z_idx],
                    c="red",
                    s=100,
                    label="Best individual",
                )

                # Line to x-y plane
                ax3.plot(
                    [best_point[x_idx], best_point[x_idx]],
                    [best_point[y_idx], best_point[y_idx]],
                    [best_point[z_idx], 0],
                    "r--",
                    alpha=0.7,
                )

                # Line to x-z plane
                ax3.plot(
                    [best_point[x_idx], best_point[x_idx]],
                    [best_point[y_idx], 1],
                    [best_point[z_idx], best_point[z_idx]],
                    "r--",
                    alpha=0.7,
                )

                # Line to y-z plane
                ax3.plot(
                    [best_point[x_idx], 0],
                    [best_point[y_idx], best_point[y_idx]],
                    [best_point[z_idx], best_point[z_idx]],
                    "r--",
                    alpha=0.7,
                )

                ax3.set_title("Pareto Front")
                ax3.set_xlabel(f"{self.objectives[x_idx]['name']}")
                ax3.set_ylabel(f"{self.objectives[y_idx]['name']}")
                ax3.set_zlabel(f"{self.objectives[z_idx]['name']}")
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
                ax3.set_zlim(0, 1)
                ax3.legend()

            else:
                print(
                    f"Oops. {self.num_objectives} objective Pareto front visualisation not suported!"
                )

        plt.suptitle("NSGA-II Optimisation Results", fontsize=16)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        plt.savefig(f"nsga2_evolution_history.png", dpi=300)
        plt.close()

    def _reset_optimiser(self):
        """
        Reset the optimiser state to prepare for a new run
        """

        super()._reset_optimiser()
        self.fitness_matrix = np.array([], dtype=float)
        self.crowding_distances = np.array([], dtype=float)
        self.fronts = []
