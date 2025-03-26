import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from gpqc.core.GPQC import GPQC
from gpqc.representations.Tree.QTree import QTree


class TestGPQC(unittest.TestCase):
    def setUp(self):
        """Set up test parameters used across tests"""
        self.gate_set = {
            "H": 0.2,
            "X": 0.1,
            "Y": 0.1,
            "Z": 0.1,
            "S": 0.1,
            "CX": 0.2,
            "Rx": 0.2,
        }

        # Create a simple requirements dictionary
        self.requirements = {
            "gp_behaviour": {
                "population_size": 10,
                "termination_params": {"max_generations": 5, "fitness_threshold": 0.95},
                "crossover_rate": 0.7,
                "mutation_rate": 0.3,
                "selection_params": {
                    "selection_method": "tournament",
                    "tournament_size": 3,
                },
            },
            "circuit_behaviour": {
                "qubits": 2,
                "max_depth": 3,
                "gates": self.gate_set,
            },
        }

        # Create a simple target circuit for testing
        target_circuit = QuantumCircuit(2)
        target_circuit.h(0)
        target_circuit.cx(0, 1)
        target_circuit.h(0)
        self.target_matrix = Operator(target_circuit).data
        self.requirements["circuit_behaviour"]["target_matrix"] = self.target_matrix

    def test_initialisation(self):
        """Test proper initialisation of GPQC"""
        gp = GPQC(self.requirements)

        self.assertEqual(gp.population_size, 10)
        self.assertEqual(gp.max_generations, 5)
        self.assertEqual(gp.fitness_threshold, 0.95)
        self.assertEqual(gp.num_qubits, 2)
        self.assertEqual(gp.max_depth, 3)
        self.assertEqual(gp.crossover_rate, 0.7)
        self.assertEqual(gp.mutation_rate, 0.3)
        self.assertEqual(gp.tournament_size, 3)
        self.assertEqual(gp.selection_method, "tournament")

        self.assertEqual(gp.current_generation, 0)
        self.assertEqual(gp.best_fitness, 0.0)
        self.assertIsNone(gp.best_individual)
        self.assertEqual(len(gp.best_fitness_history), 0)
        self.assertEqual(len(gp.avg_fitness_history), 0)

    def test_initialise_population(self):
        """Test population initialisation creates correct population"""
        gp = GPQC(self.requirements)
        gp.initialise_population(gp.population_size)

        self.assertEqual(len(gp.population), 10)

        for individual in gp.population:
            self.assertIsInstance(individual, QTree)
            self.assertEqual(individual.num_qubits, 2)
            self.assertEqual(individual.max_depth, 3)
            self.assertIsNotNone(individual.qiskit_circuit)
            self.assertIsNotNone(individual.unitary_matrix)

    def test_evaluate_population(self):
        """Test population evaluation calculates fitness scores correctly"""
        gp = GPQC(self.requirements)
        gp.initialise_population(gp.population_size)
        gp.evaluate_population()

        self.assertEqual(len(gp.fitness_scores), 10)

        for score in gp.fitness_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_select_parents_tournament(self):
        """Test tournament selection returns correct number of parents"""
        gp = GPQC(self.requirements)
        gp.initialise_population(gp.population_size)
        gp.evaluate_population()
        parents = gp.select_parents()

        self.assertEqual(len(parents), gp.population_size)

        for parent in parents:
            self.assertIsInstance(parent, QTree)

    def test_prepare_gate_set(self):
        """Test gate set preparation returns valid gates and probabilities"""
        gp = GPQC(self.requirements)

        gates, probs = gp._prepare_gate_set(self.gate_set)

        self.assertIn("H", gates)
        self.assertIn("X", gates)
        self.assertIn("CX", gates)

        self.assertAlmostEqual(np.sum(probs), 1.0)

        with self.assertRaises(ValueError):
            gp._prepare_gate_set({"InvalidGate": 1.0})

    def test_calculate_adaptive_rate(self):
        """Test adaptive rate calculation changes with generation"""
        gp = GPQC(self.requirements)

        gp.current_generation = 0
        gp._calculate_adaptive_rate()
        initial_rate = gp.adaptive_rate
        self.assertGreater(initial_rate, 0)

        gp.current_generation = 2
        gp._calculate_adaptive_rate()
        middle_rate = gp.adaptive_rate

        gp.current_generation = 4
        gp._calculate_adaptive_rate()
        final_rate = gp.adaptive_rate

        self.assertGreater(initial_rate, middle_rate)
        self.assertGreater(middle_rate, final_rate)

    def test_get_best_circuit(self):
        """Test best circuit retrieval works correctly"""
        gp = GPQC(self.requirements)

        self.assertIsNone(gp.get_best_circuit())

        gp.initialise_population(gp.population_size)
        gp.best_individual = gp.population[0]

        best_circuit = gp.get_best_circuit()
        self.assertIsInstance(best_circuit, QuantumCircuit)
        self.assertEqual(best_circuit.num_qubits, 2)

    def test_alternate_representation_class(self):
        """Test GPQC works with different representation classes"""

        class MockRepresentation(QTree):
            pass

        requirements = self.requirements.copy()
        gp = GPQC(requirements, repr_class=MockRepresentation)
        gp.initialise_population(gp.population_size)

        for individual in gp.population:
            self.assertIsInstance(individual, MockRepresentation)
