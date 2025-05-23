import matplotlib.pyplot as plt
import numpy as np
from qiskit import qpy, QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import Operator, random_statevector, process_fidelity
from qiskit.circuit.library import QFTGate, StatePreparation
from qiskit_aer import AerSimulator


def generate_gaussian_like_amplitudes(num_qubits=3) -> np.array:
    # Generate amplitudes with Gaussian distribution
    num_states = 2**num_qubits
    indices = np.arange(num_states)
    center = num_states // 2

    width_factor = num_states / 8
    amplitudes = np.exp(-((indices - center)**2) / (2 * width_factor**2))

    normalised_amplitudes = amplitudes / np.linalg.norm(amplitudes)
    return normalised_amplitudes


def create_gaussian_state_preparation_circuit(num_qubits=3) -> QuantumCircuit:
    # Create a quantum circuit using StatePreparation gate
    amplitudes = generate_gaussian_like_amplitudes(num_qubits)

    qc = QuantumCircuit(num_qubits)
    state_prep = StatePreparation(amplitudes)
    qc.append(state_prep, range(num_qubits))

    return qc


def prepare_target_circuits() -> dict[QuantumCircuit]:
    # Quantum Half Adder
    adder_circuit = QuantumCircuit(4, 2)
    adder_circuit.cx(0, 2)
    adder_circuit.cx(1, 2)
    adder_circuit.ccx(0, 1, 3)

    # Gaussian State Preparation
    gaussian_circuit = create_gaussian_state_preparation_circuit()
    gaussian_circuit = transpile(gaussian_circuit, basis_gates=[
                                 'rx', 'ry', 'rz', 'cx'])

    # Quantum Fourier Transform
    qft_circuit = QuantumCircuit(5)
    qft_circuit.append(QFTGate(5), list(range(5)))
    qft_circuit = qft_circuit.decompose()

    return {
        "adder": adder_circuit,
        "gaussian": gaussian_circuit,
        "qft": qft_circuit
    }


def random_initialise_circuits(circuits: list, input_qubits: int, measurement_bits: int) -> list[QuantumCircuit]:
    # Qiskit doesn't have a function to initialise states after
    # adding gates to a circuit, so we improvise :)
    random_vector = random_statevector(2**input_qubits)
    initialised_circuits = []

    for circuit in circuits:
        initialised = QuantumCircuit(input_qubits, measurement_bits)
        initialised.initialize(random_vector)
        initialised.barrier()
        for instruction in circuit.data:
            initialised.append(
                instruction.operation, instruction.qubits, instruction.clbits
            )
        initialised_circuits.append(initialised)

    return initialised_circuits


def get_simulation_results(circuits: list, shots=4096, method="automatic") -> list[dict]:
    simulator = AerSimulator(method=method)
    simulation_results = []

    # Get simulation state counts for each circuit
    for circuit in circuits:
        transpiled_circuit = transpile(circuit, simulator)
        counts = simulator.run(
            transpiled_circuit, shot=shots).result().get_counts()
        simulation_results.append(counts)

    return simulation_results


def plot_circuit_distributions(results: list[dict], names: list[str], shots=4096) -> None:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Collect all possible bitstrings
    all_bitstrings = set()
    for result in results:
        all_bitstrings.update(result.keys())
    all_bitstrings = sorted(list(all_bitstrings))

    plt.figure(figsize=(6, 4), dpi=300)
    x = np.arange(len(all_bitstrings))
    width = 0.8 / len(results)

    # Plot bars for each circuit
    for i, result in enumerate(results):
        values = [result.get(bitstring, 0) /
                  shots for bitstring in all_bitstrings]
        position = x + (i - len(results)/2 + 0.5) * width
        bars = plt.bar(
            position,
            values,
            width,
            label=names[i],
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )

        if len(all_bitstrings) <= 5:
            for j, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )

    fontsize = 10 if len(all_bitstrings) < 10 else (
        8 if len(all_bitstrings) < 16 else 6
    )
    plt.xlabel('Quantum States', fontsize=8, fontweight='bold')
    plt.ylabel('Probability', fontsize=12, fontweight='bold')
    plt.title('Circuit Simulation Distribution',
              fontsize=14, fontweight='bold')

    plt.xticks(x, all_bitstrings, rotation=90 if len(all_bitstrings) > 8 else 0,
               ha='right' if len(all_bitstrings) > 8 else 'center', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylim(0, 1.0)

    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.legend(fontsize=10, frameon=True, framealpha=0.9, loc='best')

    if len(all_bitstrings) > 16:
        plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(pad=1.2)

    plt.show()


def plot_resource_graphs(circuits: list[dict], names: list[str]):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Calculate metrics compared to the target circuit (index 0)
    target_statevector = Operator.from_circuit(circuits[0])
    fidelities = []

    for circuit in circuits:
        circuit_statevector = Operator.from_circuit(circuit)
        fidelity = process_fidelity(circuit_statevector, target_statevector)
        fidelities.append(fidelity)

    gate_counts = [len(circuit.data) for circuit in circuits]
    depths = [circuit.depth() for circuit in circuits]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    width = 0.7

    # Fidelities
    bars = axs[0].bar(names, fidelities, width=width, color=colors[:len(circuits)],
                      alpha=0.8, edgecolor='black', linewidth=1)

    for bar in bars:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    axs[0].set_xlabel('Circuit', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    axs[0].set_title('Circuit Fidelity', fontsize=14, fontweight='bold')
    axs[0].set_ylim(0, 1.1)  # Fidelity ranges from 0 to 1
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.6, axis='y')

    # Gate Counts
    bars = axs[1].bar(names, gate_counts, width=width, color=colors[:len(circuits)],
                      alpha=0.8, edgecolor='black', linewidth=1)

    for bar in bars:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    axs[1].set_xlabel('Circuit', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Gate Count', fontsize=12, fontweight='bold')
    axs[1].set_title('Circuit Gate Count', fontsize=14, fontweight='bold')
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.6, axis='y')
    max_count = max(gate_counts)
    min_count = min(gate_counts)
    tick_step = max(1, int((max_count - min_count) / 5))
    y_ticks = range(0, max_count + tick_step + 1, tick_step)
    axs[1].set_yticks(y_ticks)
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Circuit Depths
    bars = axs[2].bar(names, depths, width=width, color=colors[:len(circuits)],
                      alpha=0.8, edgecolor='black', linewidth=1)

    for bar in bars:
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    axs[2].set_xlabel('Circuit', fontsize=12, fontweight='bold')
    axs[2].set_ylabel('Circuit Depth', fontsize=12, fontweight='bold')
    axs[2].set_title('Circuit Depth', fontsize=14, fontweight='bold')
    axs[2].tick_params(axis='both', which='major', labelsize=10)
    axs[2].grid(True, linestyle='--', alpha=0.6, axis='y')
    max_depth = max(depths)
    min_depth = min(depths)
    tick_step = max(1, int((max_depth - min_depth) / 5))
    y_ticks = range(0, max_depth + tick_step + 1, tick_step)
    axs[2].set_yticks(y_ticks)
    axs[2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout(pad=1.2)
    plt.show()


def main() -> None:

    required_qubits = 4
    measurement_bits = 2
    plot_labels = ["Target", "Generated"]

    target_circuits = prepare_target_circuits()
    target_circuit = target_circuits["adder"].copy()

    # Get generated circuit and prepare measurement qubits
    with open("generated_simple_circuit.qpy", "rb") as file:
        generated_circuit = qpy.load(file)[0]

    plot_resource_graphs([target_circuit, generated_circuit], plot_labels)

    # Add measurements
    generated_circuit.add_register(ClassicalRegister(measurement_bits, 'c'))
    generated_circuit.barrier()
    generated_circuit.measure(2, 0)
    generated_circuit.measure(3, 1)

    target_circuit.barrier()
    target_circuit.measure(2, 0)
    target_circuit.measure(3, 1)

    # Simulate and plot probability distributions
    initialised_circuits = random_initialise_circuits(
        [target_circuit, generated_circuit], required_qubits, measurement_bits
    )

    results = get_simulation_results(initialised_circuits)
    plot_circuit_distributions(results, plot_labels)


if __name__ == "__main__":
    main()
