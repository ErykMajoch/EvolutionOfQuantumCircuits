# GPQC: Genetic Programming for Quantum Circuit Design

GPQC is a framework developed for gathering experimental results for my Bachelor's dissertation project that uses evolutionary algorithms to automatically design and optimise quantum circuits. It addresses the challenges of manual quantum circuit design by leveraging genetic programming techniques to evolve optimal circuit structures based on specified requirements.

## Overview

This project aims to automate the quantum circuit design process by:

1. Developing a complete simulation environment for quantum circuits using genetic programming
2. Implementing appropriate genetic operations for circuit evolution
3. Defining fitness functions that guide the evolution process toward desired circuit behaviour
4. Analysing the effectiveness of evolutionary optimisation compared to traditional design approaches

## Features

- Supports both single-objective and multi-objective optimisation algorithms
- Offers customisable gate sets with weighted probabilities
- Allows control over various evolutionary parameters
- Provides visualisation tools for comparing generated circuits with target circuits
- Exports circuits in standard Qiskit format for integration with other quantum tools

## Supported Circuit Types

The following quantum circuits are currently supported out-of-the-box:

1. **Quantum Half Adder** (`adder.pickle`) - Basic arithmetic circuit that adds two qubits
2. **Gaussian State Preparation** (`gaussian.pickle`) - Circuit preparing states with Gaussian-like amplitude distribution
3. **Quantum Fourier Transform** (`qft.pickle`) - Fundamental transformation for many quantum algorithms

## Usage

### Generating Circuits

You can generate quantum circuits using either single-objective or multi-objective algorithms as shown in the example below:

```python
# evolve.py
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator
from gpqc.core.GPQC import GPQC

# Define genetic algorithm parameters
ga_params = {
    "ga_algorithm": "simple",  # Use "simple" for single-objective or "nsga2" for multi-objective
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
        "qubits": 4,
        "max_depth": 3,
        "target_matrix": target_circuit_matrix,  # Your target unitary matrix
        "gates": {
            "H": 0.2,
            "X": 0.1,
            "CX": 0.1,
            # Additional gates with weights...
        },
    },
    # Only needed for multi-objective algorithm
    "nsga_params": {
        "objectives": [
            {"name": "fidelity", "minimise": False},
            {"name": "gate_count", "minimise": True},
            {"name": "circuit_depth", "minimise": True},
        ]
    }
}

# Run evolutionary algorithm
evolutionary_algorithm = GPQC(ga_params)
evolutionary_algorithm.optimise(plot_results=True)
generated_circuit = evolutionary_algorithm.get_best_circuit()

# Save generated circuit
with open("generated_circuit.qpy", "wb") as file:
    qpy.dump(generated_circuit, file)
```

### Visualising Circuit Performance

You can visualise and compare the performance of generated circuits with target circuits:

```python
# visualise.py
from qiskit import qpy, QuantumCircuit, ClassicalRegister
from visualise import plot_resource_graphs, plot_circuit_distributions, get_simulation_results, random_initialise_circuits

# Load target and generated circuits
target_circuit = ...  # Your target circuit
with open("generated_circuit.qpy", "rb") as file:
    generated_circuit = qpy.load(file)[0]

# Compare metrics (fidelity, gate count, depth)
plot_resource_graphs([target_circuit, generated_circuit], ["Target", "Generated"])

# Add measurement registers if needed

# Compare output distributions
initialised_circuits = random_initialise_circuits(
    [target_circuit, generated_circuit], 
    input_qubits=4, 
    measurement_bits=measurement_bits
)
results = get_simulation_results(initialised_circuits)
plot_circuit_distributions(results, ["Target", "Generated"])
```

## Requirements

- Python 3.8+
- NumPy 2.2+
- Matplotlib 3.10+
- Qiskit 1.4+
- Qiskit-Aer 0.16+
- Treelib 1.7+

## Installation

1. Clone Repository

```bash
git clone https://github.com/ErykMajoch/EvolutionOfQuantumCircuits.git
cd EvolutionOfQuantumCircuits
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
# Linux/MacOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate
```

3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Verify Installation

To verify the installation works correctly, you can run:

```bash
# Generate a simple quantum circuit
python generate.py

# Visualise the results
python visualise.py
```
