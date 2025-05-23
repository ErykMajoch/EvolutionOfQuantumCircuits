import pickle

from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

from gpqc.core.GPQC import GPQC


QUBUT_NUMER = 4
MAX_CIRCUIT_DEPTH = 3

GATE_SET = {
    "H": 0.2,
    "X": 0.1,
    "Y": 0.1,
    "Z": 0.1,
    "S": 0.1,
    "T": 0.05,
    "CX": 0.1,
    "CZ": 0.05,
    "Rx": 0.1,
    "Ry": 0.05,
    "Rz": 0.05,
    "CCX": 0.2,
}


def main() -> None:

    # Load target unitary matrix and create target circuit
    with open("adder.pickle", "rb") as handle:
        target_circuit_matrix = pickle.load(handle)

    target_circuit = QuantumCircuit(QUBUT_NUMER)
    target_operator = Operator(target_circuit_matrix)
    target_circuit.append(target_operator, list(range(QUBUT_NUMER)))

    print("Target circuit:")
    print(target_circuit)

    # Define genetic algorithm parameters
    ga_params = {
        "ga_algorithm": "simple",
        "ga_behaviour": {
            "population_size": 150,
            "termination_params": {
                "max_generations": 100,
                "fitness_threshold": 0.99,
            },
            "crossover_rate": 0.7,
            "mutation_rate": 0.3,
            "selection_params": {
                "selection_method": "tournament",
                "tournament_size": 3,
            },
        },
        "circuit_behaviour": {
            "qubits": QUBUT_NUMER,
            "max_depth": MAX_CIRCUIT_DEPTH,
            "target_matrix": target_circuit_matrix,
            "gates": GATE_SET,
        },
        "nsga_params": {
            "objectives": [
                {"name": "fidelity", "minimise": False},
                {"name": "gate_count", "minimise": True},
                {"name": "circuit_depth", "minimise": True},
            ]
        }
    }

    # Generate circuit using single-objective algorithm
    evolutionary_algorithm = GPQC(ga_params)
    evolutionary_algorithm.optimise(plot_results=True)
    generated_simple_circuit = evolutionary_algorithm.get_best_circuit()

    print("Generated single-obhjective circuit:")
    print(generated_simple_circuit)

    # Save genereated single-objective algorithm circuit to disk
    with open("generated_simple_circuit.qpy", "wb") as file:
        qpy.dump(generated_simple_circuit, file)

    # Switch to multi-objective algorithm
    ga_params["ga_algorithm"] = "nsga2"

    # Generate circuit using multi-objective algorithm
    evolutionary_algorithm = GPQC(ga_params)
    evolutionary_algorithm.optimise(plot_results=True)
    generated_nsga_circuit = evolutionary_algorithm.get_best_circuit()

    print("Generated multi-objective circuit:")
    print(generated_nsga_circuit)

    # Save genereated single-objective algorithm circuit to disk
    with open("generated_nsga_circuit.qpy", "wb") as file:
        qpy.dump(generated_nsga_circuit, file)


if __name__ == "__main__":
    main()
