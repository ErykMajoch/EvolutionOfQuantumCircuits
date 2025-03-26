import random
from copy import deepcopy
from typing import Optional, Tuple, Set, Dict, Any

import numpy as np
import treelib
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from gpqc.representations.Base import CircuitRepresentation
from gpqc.representations.Gates import QISKIT_GATES
from gpqc.representations.Tree.QNode import QNode


class QTree(CircuitRepresentation):
    """
    Tree-based representation of a quantum circuit.

    Attributes:
        nodes: 2D array of QNode objects representing gates in the circuit
    """

    def __init__(
        self,
        num_qubits: int,
        max_depth: int,
        gate_set: Dict[str, Any],
        gate_probs: np.array,
        random_gate_prob: float = 0.7,
    ):
        """
        Initialise a tree-based quantum circuit representation.

        Args:
            num_qubits: Number of qubits in the circuit
            max_depth: Maximum depth of the circuit
            gate_set: Dictionary mapping gate names to their selection weights
            random_gate_prob: Probability of adding a gate at each position
        """
        super().__init__(num_qubits, max_depth, gate_set, gate_probs, random_gate_prob)
        self.nodes = np.ndarray((num_qubits, max_depth), dtype=QNode)

    def generate_random_circuit(self) -> None:
        """
        Generate a random quantum circuit
        """
        self.nodes = np.ndarray((self.num_qubits, self.max_depth), dtype=QNode)

        for qubit in range(self.num_qubits):
            for depth in range(self.max_depth):
                if np.random.random() < self.random_gate_prob:
                    self._replace_gate(qubit, depth, self._generate_random_node(qubit))

        self._validate_and_fix_circuit()

    def mutate(
        self,
        mutation_rate: float,
        mutation_type: str,
        generation: int,
        max_generations: int,
        max_mutations: int = 1,
    ) -> None:
        """
        Apply mutation operations to the circuit

        Args:
            mutation_rate: Probability of applying mutation
            mutation_type: Type of mutation operation ('replace', 'parameter', or 'delete')
            generation: Current generation number
            max_generations: Maximum number of generations
            max_mutations: Maximum number of mutations to apply
        """

        for _ in range(max_mutations):
            if np.random.random() > mutation_rate:
                continue
            qubit = np.random.randint(0, self.num_qubits)
            depth = np.random.randint(0, self.max_depth)
            match mutation_type:
                case "replace":
                    new_node = self._generate_random_node(qubit)
                    self._replace_gate(qubit, depth, new_node)
                case "parameter":
                    node = self._get_node(qubit, depth)
                    if node:
                        if angle := node.params.get("angle", None):
                            adjustment = (
                                1 - generation / max_generations
                            ) * np.random.uniform(-np.pi / 2, np.pi / 2)
                            node.params["angle"] = (angle + adjustment) % (2 * np.pi)
                            self._replace_gate(qubit, depth, node)
                case "delete":
                    self._replace_gate(qubit, depth, None)
                case _:
                    raise ValueError(
                        f"Invalid mutation type: {mutation_type}. Supported types are 'replace', 'parameter' and 'delete'"
                    )
        self._validate_and_fix_circuit()

    def crossover(self, other: "QTree", crossover_rate: float = 0.7) -> "QTree":
        """
        Create a new circuit by combining this circuit with another

        Args:
            other: Another QTree circuit to crossover with
            crossover_rate: Probability of performing crossover

        Returns:
            A new QTree resulting from the crossover
        """
        offspring = self.replicate()

        if np.random.random() > crossover_rate:
            return offspring

        qubit = np.random.randint(0, self.num_qubits)
        depth = np.random.randint(0, self.max_depth)

        offspring.nodes[qubit, depth:] = other.nodes[qubit, depth:]
        self._validate_and_fix_circuit()

        return offspring

    def replicate(self) -> "QTree":
        """
        Create a deep copy of this circuit

        Returns:
            A new identical copy of this QTree
        """
        return deepcopy(self)

    def calculate_similarity(self, other: "QTree") -> float:
        """
        Calculate similarity between this circuit and another circuit

        Args:
            other: Another QTree object to compare with

        Returns:
            Float value between 0.0 and 1.0, where 1.0 means identical circuits
            and 0.0 means completely different circuits.
        """
        if not isinstance(other, QTree):
            raise TypeError(f"Expected {type(self)} instance, got {type(other)}")

        if self.num_qubits != other.num_qubits or self.max_depth != other.max_depth:
            return 0.0

        matched_positions = 0
        total_positions = 0

        for qubit in range(self.num_qubits):
            for depth in range(self.max_depth):
                total_positions += 1

                node_self = self._get_node(qubit, depth)
                node_other = other._get_node(qubit, depth)

                # Check if same gate exists at same position
                if (node_self is None and node_other is None) or (
                    node_self is not None
                    and node_other is not None
                    and node_self.gate_type == node_other.gate_type
                ):
                    matched_positions += 1

        if total_positions > 0:
            return matched_positions / total_positions
        return 0.0

    def _to_qiskit(self) -> QuantumCircuit:
        """
        Convert this circuit representation to a Qiskit QuantumCircuit object

        Returns:
            Qiskit QuantumCircuit representing this circuit
        """
        circuit = QuantumCircuit(self.num_qubits)

        for depth in range(self.max_depth):
            layer = self.nodes[:, depth].T
            for node in layer:
                if node:
                    gate = QISKIT_GATES[node.gate_type]
                    if node.gate_category in ["phase", "rotation"]:
                        gate = gate(node.params.get("angle"))
                    args = node.params.get("control_qubits", []) + node.targets
                    circuit.append(gate, args)
        return circuit

    def _replace_gate(self, qubit: int, depth: int, node: QNode | None) -> None:
        """
        Replace a gate at the specified position in the circuit

        Args:
            qubit: Qubit index
            depth: Circuit depth position
            node: QNode to place at the position, or None to remove gate
        """

        if not (0 <= qubit < self.num_qubits):
            raise IndexError(
                f"Qubit index {qubit} out of range [0, {self.num_qubits - 1}]"
            )

        if not (0 <= depth < self.max_depth):
            raise IndexError(f"Depth {depth} out of range [0, {self.max_depth - 1}]")

        self.nodes[qubit, depth] = node

    def _generate_random_node(self, qubit_index: int) -> QNode:
        """
        Generate a random QNode for the specified qubit

        Args:
            qubit_index: Index of the qubit

        Returns:
            A randomly generated QNode
        """
        gate_name = np.random.choice(list(self.gate_set.keys()), p=self.gate_probs)
        gate_info = self.gate_set[gate_name]

        targets = [qubit_index]
        params = None
        category = gate_info["category"]

        match category:
            case "rotation" | "phase":
                params = {"angle": np.random.uniform(0, 2 * np.pi)}
            case "controlled":
                available_qubits = [
                    q for q in range(self.num_qubits) if q != qubit_index
                ]
                control_qubits = np.random.choice(
                    available_qubits,
                    size=gate_info["num_qubits"] - 1,
                    replace=False,
                ).tolist()
                params = {"control_qubits": control_qubits}
            case "swap":
                available_qubits = [
                    q for q in range(self.num_qubits) if q != qubit_index
                ]
                swap_qubit = np.random.choice(available_qubits, replace=False).tolist()
                targets.append(swap_qubit)

        return QNode(gate_name, category, targets, params)

    def _validate_and_fix_circuit(self) -> None:
        """
        Validate the circuit and fix any inconsistencies

        This method ensures that controlled gates and multi-qubit gates
        don't conflict with other gates on the same qubits at the same depth.
        """
        checked_nodes = set()
        dependent_nodes = self._get_dependent_nodes()

        while to_check := dependent_nodes - checked_nodes:
            current_node, depth = random.choice(list(to_check))

            if (current_node, depth) in checked_nodes:
                continue

            # Clear gates between controlled gates
            stationary = current_node.targets[0]
            qubits_to_process = []

            if len(current_node.targets) > 1:
                qubits_to_process.append(current_node.targets[1:])
            qubits_to_process.extend(current_node.params.get("control_qubits"))

            if min(qubits_to_process) < stationary:
                start_index = min(qubits_to_process)
                end_index = stationary
            else:
                start_index = stationary + 1
                end_index = max(qubits_to_process) + 1

            for qubit in range(start_index, end_index):
                self._replace_gate(qubit, depth, None)

            checked_nodes.add((current_node, depth))
            dependent_nodes = self._get_dependent_nodes()

        self.qiskit_circuit = self._to_qiskit()
        self.unitary_matrix = Operator(self.qiskit_circuit).data

    def _get_node(self, qubit: int, depth: int) -> Optional[QNode]:
        """
        Get the node at the specified position

        Args:
            qubit: Qubit index
            depth: Circuit depth position

        Returns:
            QNode at the specified position, or None if empty
        """
        if not (0 <= qubit < self.num_qubits):
            raise IndexError(
                f"Qubit index {qubit} out of range [0, {self.num_qubits - 1}]"
            )

        if not (0 <= depth < self.max_depth):
            raise IndexError(f"Depth {depth} out of range [0, {self.max_depth - 1}]")

        return self.nodes[qubit, depth]

    def _get_dependent_nodes(self) -> Set[Tuple]:
        """
        Get all nodes that depend on multiple qubits

        Returns:
            Set of tuples (node, depth) for nodes that are controlled gates or swaps
        """
        dependent_nodes = set()
        for q in range(self.num_qubits):
            for d in range(self.max_depth):
                node = self._get_node(q, d)
                if node is not None and node.gate_category in ["controlled", "swap"]:
                    dependent_nodes.add((node, d))

        return dependent_nodes

    def __repr__(self):
        """
        Create a string representation of the circuit as a tree

        Returns:
            String representation of the circuit
        """
        tree = treelib.Tree()
        tree.create_node("Quantum Circuit", "root")

        for q in range(self.num_qubits):
            qubit_id = f"q{q}"
            tree.create_node(f"Qubit {q}", qubit_id, parent="root")

            parent_id = qubit_id
            for d in range(self.max_depth):
                depth_id = f"q{q}_d{d}"
                node = self._get_node(q, d)

                if node is not None:
                    label = f"Depth {d + 1}: {node.gate_type}"
                    if node.params:
                        param_str = ", ".join(
                            f"{k}={v}" for k, v in node.params.items()
                        )
                        label += f" ({param_str})"
                    tree.create_node(label, depth_id, parent=parent_id)
                else:
                    tree.create_node(f"Depth {d + 1}: None", depth_id, parent=parent_id)

                parent_id = depth_id

        return tree.show(stdout=False)
