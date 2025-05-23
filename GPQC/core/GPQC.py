from typing import Dict, Type, Optional

from qiskit import QuantumCircuit

from gpqc.optimisers.GAOptimiser import GAOptimiser
from gpqc.optimisers.NSGA2Optimiser import NSGA2Optimiser
from gpqc.representations.CircuitRepresentation import CircuitRepresentation
from gpqc.representations.Tree.QTree import QTree


class GPQC:
    def __init__(
            self, requirements: Dict, representation: Type[CircuitRepresentation] = QTree
    ) -> None:
        """
        Initialise the GPQC optimiser with specified requirements and representation

        Args:
            requirements: Dictionary containing parameters for the optimisation process
            representation: Class defining how quantum circuits are represented in the GA
        """

        self.algorithm_type = requirements.get("ga_algorithm", "simple")

        match self.algorithm_type:
            case "simple":
                self.algorithm = GAOptimiser(requirements, representation)
            case "nsga2":
                self.algorithm = NSGA2Optimiser(requirements, representation)
            case _:
                raise ValueError(
                    f"Unknown GA algorithm: {self.algorithm_type}")

    def optimise(self, plot_results=False) -> None:
        """
        Run the optimisation process

        Args:
            plot_results: Whether to generate plots of the optimisation results
        """

        print(f"Using the {self.algorithm_type} algorithm!")
        self.algorithm.run(plot_results)

    def get_best_circuit(self) -> Optional[QuantumCircuit]:
        """
        Retrieve the best quantum circuit found during optimisation.

        Returns:
            The best quantum circuit if one was found, otherwise None
        """

        return self.algorithm.get_best_circuit()
