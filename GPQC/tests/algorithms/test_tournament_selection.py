import unittest

import numpy as np

from gpqc.algorithms.selection import tournament_selection


class TestTournamentSelection(unittest.TestCase):
    def test_empty_population(self):
        """Test if an empty population returns an empty selection"""
        population = np.array([])
        fitness_scores = np.array([])
        result = tournament_selection(population, fitness_scores, 0, 100, 2)
        self.assertEqual(len(result), 0)

    def test_mismatched_population_and_fitness_scores(self):
        """Test if ValueError is raised when population and fitness_scores have different lengths"""
        population = np.array([[1, 2], [3, 4], [5, 6]])
        fitness_scores = np.array([0.5, 0.7])
        with self.assertRaises(ValueError):
            tournament_selection(population, fitness_scores, 0, 100, 2)

    def test_tournament_size_calculation(self):
        """Test that tournament size is calculated correctly based on generation"""
        np.random.seed(42)
        population = np.array([[i, i] for i in range(10)])
        fitness_scores = np.array([i for i in range(10)])

        for generation in [0, 50, 100]:
            result = tournament_selection(
                population, fitness_scores, generation, 100, 3
            )
            self.assertEqual(len(result), len(population))

    def test_selection_bias(self):
        """Test that selection favors individuals with higher fitness scores"""
        np.random.seed(42)
        population = np.array([[i, i] for i in range(5)])
        fitness_scores = np.array([1, 1, 1, 1, 10])

        selections_count = {i: 0 for i in range(5)}
        for _ in range(100):
            selected = tournament_selection(population, fitness_scores, 0, 100, 2)
            for individual in selected:
                idx = individual[0]
                selections_count[idx] += 1

        for index in range(0, 4):
            self.assertGreater(selections_count[4], selections_count[index])

    def test_selection_preserves_shape(self):
        """Test that selection preserves the shape of individuals"""
        np.random.seed(42)
        population = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
        fitness_scores = np.array([0.5, 0.7, 0.9])

        result = tournament_selection(population, fitness_scores, 0, 100, 2)

        self.assertEqual(result.shape[1:], population.shape[1:])
        self.assertEqual(len(result), len(population))

    def test_tournament_size_limits(self):
        """Test that tournament size respects minimum and maximum limits"""
        np.random.seed(42)
        population = np.array([[i, i] for i in range(10)])
        fitness_scores = np.array([i for i in range(10)])

        for tournament_size in [1, 20]:
            result = tournament_selection(
                population, fitness_scores, 0, 100, tournament_size
            )
            self.assertEqual(len(result), len(population))
