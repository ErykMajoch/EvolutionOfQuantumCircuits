import unittest
from copy import deepcopy

import numpy as np
from qiskit.circuit import QuantumCircuit

from gpqc.representations.Tree.QNode import QNode
from gpqc.representations.Tree.QTree import QTree


class TestQTree(unittest.TestCase):
    def setUp(self):
        """Set up test parameters used across tests"""
        self.num_qubits = 3
        self.max_depth = 5
        self.gate_set = {
            "H": {"category": "single", "num_qubits": 1},
            "X": {"category": "single", "num_qubits": 1},
            "Rx": {"category": "rotation", "num_qubits": 1},
            "CX": {"category": "controlled", "num_qubits": 2},
        }
        self.gate_probs = np.array([0.3, 0.2, 0.3, 0.2])
        self.random_gate_prob = 0.7

    def test_initialisation(self):
        """Test proper initialisation of QTree"""
        qtree = QTree(
            self.num_qubits,
            self.max_depth,
            self.gate_set,
            self.gate_probs,
            self.random_gate_prob,
        )

        self.assertEqual(qtree.num_qubits, self.num_qubits)
        self.assertEqual(qtree.max_depth, self.max_depth)
        self.assertEqual(qtree.random_gate_prob, self.random_gate_prob)
        self.assertEqual(qtree.nodes.shape, (self.num_qubits, self.max_depth))

    def test_generate_random_circuit(self):
        """Test random circuit generation with appropriate node population"""
        np.random.seed(42)
        qtree = QTree(
            self.num_qubits,
            self.max_depth,
            self.gate_set,
            self.gate_probs,
            self.random_gate_prob,
        )
        qtree.generate_random_circuit()

        non_empty_nodes = np.count_nonzero(qtree.nodes)
        self.assertGreater(non_empty_nodes, 0)

        for qubit in range(self.num_qubits):
            for depth in range(self.max_depth):
                node = qtree.nodes[qubit, depth]
                if node is not None:
                    self.assertIsInstance(node, QNode)

    def test_mutate_replace(self):
        """Test that gate replacement mutation changes the circuit appropriately"""
        np.random.seed(42)
        qtree = QTree(
            self.num_qubits,
            self.max_depth,
            self.gate_set,
            self.gate_probs,
            self.random_gate_prob,
        )

        qtree.generate_random_circuit()
        original_nodes = deepcopy(qtree.nodes)
        qtree.mutate(1.0, "replace", 0, 100, 1)

        self.assertFalse(np.array_equal(original_nodes, qtree.nodes))

    def test_mutate_parameter(self):
        """Test that parameter mutation changes gate parameters correctly"""
        np.random.seed(42)
        gate_set = {"Rx": {"category": "rotation", "num_qubits": 1}}
        gate_probs = np.array([1.0])
        qtree = QTree(self.num_qubits, self.max_depth,
                      gate_set, gate_probs, 1.0)
        qtree.generate_random_circuit()

        param_nodes = []
        for qubit in range(self.num_qubits):
            for depth in range(self.max_depth):
                node = qtree.nodes[qubit, depth]
                if node is not None and "angle" in node.params:
                    param_nodes.append((qubit, depth, node.params["angle"]))

        if param_nodes:
            qtree.mutate(1.0, "parameter", 0, 100, 1)

            changed = False
            for qubit, depth, original_angle in param_nodes:
                node = qtree.nodes[qubit, depth]
                if node is not None and node.params.get("angle") != original_angle:
                    changed = True
                    break

            self.assertTrue(changed)

    def test_mutate_insert(self):
        """Test that insert mutation adds gates to empty positions in the circuit"""
        np.random.seed(42)
        qtree = QTree(
            self.num_qubits, self.max_depth, self.gate_set, self.gate_probs, 0.5
        )
        # Using a lower random_gate_prob to ensure we have empty positions
        qtree.generate_random_circuit()

        non_empty_before = np.count_nonzero(qtree.nodes)

        # Apply insert mutation with high probability to ensure it happens
        qtree.mutate(1.0, "insert", 0, 100, 3)

        non_empty_after = np.count_nonzero(qtree.nodes)

        # There should be more gates after insertion
        self.assertGreater(non_empty_after, non_empty_before)

    def test_mutate_delete(self):
        """Test that delete mutation removes gates from the circuit"""
        np.random.seed(42)
        qtree = QTree(
            self.num_qubits, self.max_depth, self.gate_set, self.gate_probs, 1.0
        )
        qtree.generate_random_circuit()

        non_empty_before = np.count_nonzero(qtree.nodes)
        qtree.mutate(1.0, "delete", 0, 100, 1)
        non_empty_after = np.count_nonzero(qtree.nodes)

        self.assertLessEqual(non_empty_after, non_empty_before)

    def test_mutate_invalid_type(self):
        """Test that an invalid mutation type raises ValueError"""
        qtree = QTree(self.num_qubits, self.max_depth,
                      self.gate_set, self.gate_probs)

        with self.assertRaises(ValueError):
            qtree.mutate(1.0, "invalid_type", 0, 100, 1)

    def test_crossover(self):
        """Test that crossover creates a valid offspring combining parent circuits"""
        np.random.seed(42)
        qtree1 = QTree(self.num_qubits, self.max_depth,
                       self.gate_set, self.gate_probs)
        qtree1.generate_random_circuit()

        qtree2 = QTree(self.num_qubits, self.max_depth,
                       self.gate_set, self.gate_probs)
        qtree2.generate_random_circuit()

        offspring = qtree1.crossover(qtree2, 0)

        self.assertIsInstance(offspring, QTree)

        diff_from_parent1 = False
        diff_from_parent2 = False

        for qubit in range(self.num_qubits):
            for depth in range(self.max_depth):
                if (
                    offspring.nodes[qubit, depth] is not None
                    and qtree1.nodes[qubit, depth] is None
                ) or (
                    offspring.nodes[qubit, depth] is None
                    and qtree1.nodes[qubit, depth] is not None
                ):
                    diff_from_parent1 = True

                if (
                    offspring.nodes[qubit, depth] is not None
                    and qtree2.nodes[qubit, depth] is None
                ) or (
                    offspring.nodes[qubit, depth] is None
                    and qtree2.nodes[qubit, depth] is not None
                ):
                    diff_from_parent2 = True

        self.assertTrue(diff_from_parent1 or diff_from_parent2)

    def test_replicate(self):
        """Test that replication creates an identical but independent copy"""
        np.random.seed(42)
        qtree = QTree(self.num_qubits, self.max_depth,
                      self.gate_set, self.gate_probs)
        qtree.generate_random_circuit()

        replica = qtree.replicate()

        for qubit in range(self.num_qubits):
            for depth in range(self.max_depth):
                if qtree.nodes[qubit, depth] is None:
                    self.assertIsNone(replica.nodes[qubit, depth])
                else:
                    self.assertEqual(
                        qtree.nodes[qubit, depth].gate_type,
                        replica.nodes[qubit, depth].gate_type,
                    )

        # Verify it's a deep copy by modifying the original
        if qtree.nodes[0, 0] is None:
            qtree.nodes[0, 0] = QNode("H", "single", [0])
        else:
            qtree.nodes[0, 0] = None

        self.assertNotEqual(
            qtree.nodes[0, 0] is None, replica.nodes[0, 0] is None)

    def test_to_qiskit(self):
        """Test conversion to Qiskit circuit produces valid QuantumCircuit objects"""
        np.random.seed(42)
        qtree = QTree(self.num_qubits, self.max_depth,
                      self.gate_set, self.gate_probs)
        qtree.generate_random_circuit()

        qiskit_circuit = qtree._to_qiskit()

        self.assertIsInstance(qiskit_circuit, QuantumCircuit)
        self.assertEqual(qiskit_circuit.num_qubits, self.num_qubits)

    def test_validate_and_fix_circuit(self):
        """Test that circuit validation resolves conflicts between gates"""
        np.random.seed(42)
        gate_set = {"CX": {"category": "controlled", "num_qubits": 2}}
        gate_probs = np.array([1.0])
        qtree = QTree(self.num_qubits, self.max_depth,
                      gate_set, gate_probs, 1.0)
        qtree.generate_random_circuit()

        controlled_node = QNode("CX", "controlled", [0], {
                                "control_qubits": [1]})
        qtree.nodes[0, 0] = controlled_node
        qtree.nodes[1, 0] = QNode("H", "single", [1])  # Conflicting gate
        qtree._validate_and_fix_circuit()

        self.assertIsNone(qtree.nodes[1, 0])

    def test_replace_gate(self):
        """Test that gates can be correctly replaced or removed at specific positions"""
        qtree = QTree(self.num_qubits, self.max_depth,
                      self.gate_set, self.gate_probs)

        node = QNode("H", "single", [0])
        qtree._replace_gate(0, 0, node)
        self.assertEqual(qtree.nodes[0, 0], node)

        qtree._replace_gate(0, 0, None)
        self.assertIsNone(qtree.nodes[0, 0])

        with self.assertRaises(IndexError):
            qtree._replace_gate(self.num_qubits, 0, node)

        with self.assertRaises(IndexError):
            qtree._replace_gate(0, self.max_depth, node)

    def test_get_node(self):
        """Test that nodes can be correctly retrieved from specific positions"""
        qtree = QTree(self.num_qubits, self.max_depth,
                      self.gate_set, self.gate_probs)

        node = QNode("H", "single", [0])
        qtree.nodes[0, 0] = node

        retrieved_node = qtree._get_node(0, 0)
        self.assertEqual(retrieved_node, node)

        with self.assertRaises(IndexError):
            qtree._get_node(self.num_qubits, 0)

        with self.assertRaises(IndexError):
            qtree._get_node(0, self.max_depth)

    def test_calculate_similarity(self):
        """Test similarity calculation between partially similar circuits"""
        np.random.seed(42)
        qtree1 = QTree(self.num_qubits, self.max_depth,
                       self.gate_set, self.gate_probs)
        qtree2 = QTree(self.num_qubits, self.max_depth,
                       self.gate_set, self.gate_probs)

        for qubit in range(self.num_qubits):
            for depth in range(self.max_depth):
                if depth < self.max_depth // 2:
                    node = QNode("H", "single", [qubit])
                    qtree1._replace_gate(qubit, depth, node)
                    qtree2._replace_gate(qubit, depth, node)
                else:
                    qtree1._replace_gate(
                        qubit, depth, QNode("X", "single", [qubit]))
                    qtree2._replace_gate(
                        qubit, depth, QNode("Rx", "rotation", [
                                            qubit], {"angle": 0.5})
                    )

        similarity = qtree1.calculate_similarity(qtree2)
        self.assertGreaterEqual(similarity, 0.4)
        self.assertLessEqual(similarity, 0.6)
