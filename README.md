# Parameterized Quantum Circuit Benchmarks

This repository contains a modular benchmark for parameterized quantum circuit
optimization across combinatorial optimization, quantum chemistry, and quantum
machine learning workloads.

## Setup

The project is tested with Python 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list includes Qiskit, Qiskit Aer, Qiskit Nature, PySCF,
scikit-learn, PyTorch, and the plotting libraries used by the experiments.
Installing `requirements.txt` is required before running any experiment.

## Experiments

Run the entry points from the repository root:

```bash
python main_maxcut.py
python main_h2.py
python main_iris.py
python main_qcnn.py
```

The workloads are:

- **MaxCut:** a five-node QAOA instance with two QAOA layers.
- **Hydrogen VQE:** molecular hydrogen from `h2.xyz`, using STO-3G, Jordan-Wigner mapping, and UCCSD.
- **Iris:** binary classification of Iris classes 0 and 1 using a four-qubit, two-layer circuit.
- **MNIST QCNN:** binary classification of digits 0 and 1 using an eight-qubit QCNN.

The default experiment budget is `EPOCHS = 100`, with 256 shots.


## Outputs

Experiments write generated artifacts under `images/`, including PNG, PDF, and
SVG plots, convergence CSV files, text summaries, and LaTeX checkpoint tables.
The reporting code tracks objective history, circuit evaluations, elapsed time,
and convergence information.

MNIST data is downloaded by the QCNN problem on first use. The H2 experiment
uses the included `h2.xyz` file, so it should be run from the repository root.

## Hardware

Simulation is enabled by default. To use IBM Quantum hardware, change the
`simulation` setting in the relevant entry point and configure a local
`QiskitRuntimeService` account. All four entry points select the
`ibm_aachen` backend for hardware runs.

## Implemented Methods

The modular direction estimators include finite differences, SPSA, PGPE,
parameter-shift gradients, and the implemented diagonal objective-curvature
preconditioner (OCP). COBYLA, Nelder-Mead, and Bayesian optimization are
included as derivative-free baselines.
