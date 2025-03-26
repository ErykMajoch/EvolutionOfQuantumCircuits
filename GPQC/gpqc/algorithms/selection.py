import numpy as np


def tournament_selection(
    population: np.array,
    fitness_scores: np.array,
    generation: int,
    max_generations: int,
    base_tournament_size: int,
) -> np.array:
    """
    Select individuals from the population using tournament selection

    Args:
        population: List of individuals in the population
        fitness_scores: List of fitness scores corresponding to each individual
        generation: Current generation number
        max_generations: Maximum number of generations
        base_tournament_size: Base size of each tournament

    Returns:
        List of selected individuals for reproduction
    """

    if len(population) != len(fitness_scores):
        raise ValueError("Population and fitness scores must have the same length")

    if len(population) == 0:
        return []

    tournament_size = np.clip(
        base_tournament_size + int((generation / max_generations) * 2),
        2,
        len(population) // 2,
    )

    pop_size = len(population)
    tournament_indices = np.random.randint(
        0, pop_size, size=(pop_size, tournament_size)
    )
    tournament_fitness = np.take(fitness_scores, tournament_indices)
    winner_positions = np.argmax(tournament_fitness, axis=1)
    winner_indices = np.array(
        [tournament_indices[i, winner_positions[i]] for i in range(pop_size)]
    )

    return population[winner_indices]
