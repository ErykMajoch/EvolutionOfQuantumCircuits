from typing import Dict, Any, Optional, Type

import matplotlib.pyplot as plt
import numpy as np

from gpqc.algorithms.selection import tournament_selection
from gpqc.evaluation.fitness_functions import state_fidelity_fitness
from gpqc.optimisers.BaseOptimiser import BaseOptimiser
from gpqc.representations.CircuitRepresentation import CircuitRepresentation
from gpqc.representations.Tree.QTree import QTree


class SimpleGAOptimiser(BaseOptimiser):
    def __init__(
        self,
        requirements: Dict[str, Any],
        representation: Type[CircuitRepresentation] = QTree,
    ) -> None:
        super().__init__(requirements, representation)

        # Evolution parameters
        ga_params = requirements.get("ga_behaviour", {})
        self.diversity_threshold = ga_params.get("diversity_threshold", 0.3)

        selection_params = ga_params.get("selection_params", {})
        self.selection_method = selection_params.get("selection_method", "tournament")
        if self.selection_method == "tournament":
            self.tournament_size = selection_params.get("tournament_size", 3)

        # Optimiser parameters
        self.fitness_scores = np.array([], dtype=float)

    def evaluate_population(self, population: np.array = None) -> Optional[np.ndarray]:
        self.fitness_scores = np.ndarray((self.population_size,), dtype=float)
        for index, individual in enumerate(self.population):
            operator = individual.qiskit_operator
            self.fitness_scores[index] = state_fidelity_fitness(
                operator, self.target_operator
            )

    def calculate_diversity(self) -> float:
        diversity_sum = 0.0
        total_comparisons = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                similarity = self.population[i].calculate_similarity(self.population[j])
                diversity_sum += 1.0 - similarity
                total_comparisons += 1

        return diversity_sum / total_comparisons if total_comparisons > 0 else 0

    def apply_fitness_sharing(self) -> np.array:
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
                return np.random.choice(
                    self.population,
                    size=self.population_size,
                    p=fitness_for_selection,
                )

    def run(self, plot_results: bool = False) -> None:
        self._reset_optimiser()
        self.initialise_population(self.population_size)
        self.evaluate_population()

        for generation in range(self.max_generations):
            self.current_generation = generation

            # Handle stagnation
            if self.recently_stagnated:
                self.evaluate_population()
                self.recently_stagnated = False

            if (
                self.current_generation > 50
                and abs(self.best_fidelity - self.metrics_history["best_fitness"][-50])
                < 0.001
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

            # Perform GA operators
            adaptive_rates = self._calculate_adaptive_rates()
            parents = self.select_parents()

            elite_indices = np.argsort(self.fitness_scores)[
                -adaptive_rates["elite_count"] :
            ]
            new_population = np.array(
                [self.population[i].replicate() for i in elite_indices]
            )
            size = self.population_size - adaptive_rates["elite_count"]
            offspring = self._create_offspring(parents, size)

            self.population = np.append(new_population, offspring)
            self.evaluate_population()

            # Track progress
            best_index = np.argmax(self.fitness_scores)
            best_fitness = self.fitness_scores[best_index]

            self.metrics_history["best_fitness"].append(best_fitness)
            self.metrics_history["average_fitness"].append(np.mean(self.fitness_scores))
            self.metrics_history["crossover_rate"].append(
                adaptive_rates["crossover_rate"]
            )
            self.metrics_history["mutation_rate"].append(
                adaptive_rates["mutation_rate"]
            )
            self.metrics_history["max_mutations"].append(
                adaptive_rates["max_mutations"]
            )

            if best_fitness > self.best_fidelity:
                self.best_fidelity = best_fitness
                self.best_individual = self.population[best_index].replicate()

            # Check termination condition
            if self.best_fidelity >= self.fitness_threshold:
                print(
                    f"Reached fitness threshold at generation {self.current_generation}"
                )
                break

            # Log process
            if generation % 10 == 0:
                diversity = self.calculate_diversity()
                print(
                    f"Generation {self.current_generation}: Best fitness = {self.best_fidelity:.4f},"
                    f" Average fitness = {np.mean(self.fitness_scores):.4f},"
                    f" Diversity = {diversity:.2f}"
                )

        print(f"Evolution completed after {self.current_generation + 1} generations")
        print(f"Best fitness achieved: {self.best_fidelity:.4f}")

        if plot_results:
            self._plot_results()

    def _plot_results(self) -> None:
        plt.figure(figsize=(12, 12))

        # Plot fitness
        plt.subplot(3, 1, 1)
        plt.plot(self.metrics_history["best_fitness"], "b-", label="Best Fitness")
        plt.plot(self.metrics_history["average_fitness"], "r-", label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Evolution")
        plt.legend()
        plt.grid(True)

        # Plot fitness difference (to show exploration)
        plt.subplot(3, 1, 2)
        fitness_gap = [
            best - avg
            for best, avg in zip(
                self.metrics_history["best_fitness"],
                self.metrics_history["average_fitness"],
            )
        ]
        plt.plot(fitness_gap, "g-", label="Fitness Gap (Exploration Indicator)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Gap")
        plt.title("Exploration Indicator (Best Fitness - Average Fitness)")
        plt.grid(True)
        plt.legend()

        # Plot rates
        plt.subplot(3, 1, 3)
        plt.plot(self.metrics_history["crossover_rate"], "b-", label="Crossover Rate")
        plt.plot(self.metrics_history["mutation_rate"], "r-", label="Mutation Rate")

        # Plot max mutations on secondary y-axis for better visibility (as it's an integer)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(self.metrics_history["max_mutations"], "g--", label="Max Mutations")
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

    def _reset_optimiser(self) -> None:
        super()._reset_optimiser()
        self.fitness_scores = np.array([], dtype=float)
