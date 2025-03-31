import unittest

import numpy as np
from qiskit.quantum_info import Operator

from gpqc.evaluation.fitness_functions import state_fidelity_fitness


class TestMatrixSimilarity(unittest.TestCase):
    def setUp(self):
        """Set up test parameters used across tests"""
        self.identity_2x2 = Operator(np.eye(2))
        self.zero_2x2 = Operator(np.zeros((2, 2)))
        self.hadamard_2x2 = Operator(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        self.identity_4x4 = Operator(np.eye(4))

    def test_identical_matrices(self):
        """Test similarity of identical matrices"""
        result = state_fidelity_fitness(self.identity_2x2, self.identity_2x2)
        self.assertAlmostEqual(result, 1.0)

        result = state_fidelity_fitness(self.hadamard_2x2, self.hadamard_2x2)
        self.assertAlmostEqual(result, 1.0)

    def test_orthogonal_matrices(self):
        """Test similarity of orthogonal matrices"""
        orthogonal = Operator(np.array([[0, 1j], [1j, 0]]))
        result = state_fidelity_fitness(self.identity_2x2, orthogonal)

        self.assertAlmostEqual(result, 0.0)

    def test_partial_similarity(self):
        """Test similarity of partially similar matrices"""
        partial = Operator(np.array([[1, 0], [0, 1j]]))
        result = state_fidelity_fitness(self.identity_2x2, partial)

        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_different_dimensions(self):
        """Test matrices of different dimensions"""
        result = state_fidelity_fitness(self.identity_2x2, self.identity_4x4)
        self.assertEqual(result, 0.0)

    def test_none_input(self):
        """Test None inputs"""
        result = state_fidelity_fitness(None, self.identity_2x2)
        self.assertEqual(result, 0.0)

        result = state_fidelity_fitness(self.identity_2x2, None)
        self.assertEqual(result, 0.0)

        result = state_fidelity_fitness(None, None)
        self.assertEqual(result, 0.0)
