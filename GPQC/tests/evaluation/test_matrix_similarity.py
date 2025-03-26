import unittest

import numpy as np

from gpqc.evaluation.evaluation_functions import matrix_similarity


class TestMatrixSimilarity(unittest.TestCase):
    def setUp(self):
        """Set up test parameters used across tests"""
        self.identity_2x2 = np.eye(2)
        self.zero_2x2 = np.zeros((2, 2))
        self.hadamard_2x2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.identity_4x4 = np.eye(4)

    def test_identical_matrices(self):
        """Test similarity of identical matrices"""
        result = matrix_similarity(self.identity_2x2, self.identity_2x2)
        self.assertAlmostEqual(result, 1.0)

        result = matrix_similarity(self.hadamard_2x2, self.hadamard_2x2)
        self.assertAlmostEqual(result, 1.0)

    def test_orthogonal_matrices(self):
        """Test similarity of orthogonal matrices"""
        orthogonal = np.array([[0, 1j], [1j, 0]])
        result = matrix_similarity(self.identity_2x2, orthogonal)

        self.assertAlmostEqual(result, 0.0)

    def test_partial_similarity(self):
        """Test similarity of partially similar matrices"""
        partial = np.array([[1, 0], [0, 1j]])
        result = matrix_similarity(self.identity_2x2, partial)

        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_different_dimensions(self):
        """Test matrices of different dimensions"""
        result = matrix_similarity(self.identity_2x2, self.identity_4x4)
        self.assertEqual(result, 0.0)

    def test_none_input(self):
        """Test None inputs"""
        result = matrix_similarity(None, self.identity_2x2)
        self.assertEqual(result, 0.0)

        result = matrix_similarity(self.identity_2x2, None)
        self.assertEqual(result, 0.0)

        result = matrix_similarity(None, None)
        self.assertEqual(result, 0.0)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        invalid_input = ["not", "a", "matrix"]

        result = matrix_similarity(invalid_input, self.identity_2x2)
        self.assertEqual(result, 0.0)
