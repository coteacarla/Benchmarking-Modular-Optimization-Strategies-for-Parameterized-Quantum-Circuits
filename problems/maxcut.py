import numpy as np
import rustworkx as rx
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

from base import QuantumProblem


class MaxCutProblem(QuantumProblem):
    def __init__(self, num_nodes: int = 5, p_layers: int = 2, seed: int = 42):
        self.num_nodes = num_nodes
        self.p = p_layers
        self.graph = self._build_graph(seed)
        self.hamiltonian = self._build_hamiltonian()
        self.ansatz = QAOAAnsatz(self.hamiltonian, reps=self.p)

    def _build_graph(self, seed):
        rng = np.random.default_rng(seed)
        graph = rx.PyGraph()
        graph.add_nodes_from(np.arange(0, self.num_nodes, 1))

        edge_list = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if rng.random() > 0.5:
                    edge_list.append((i, j, 1.0))

        graph.add_edges_from(edge_list)
        return graph

    def _build_hamiltonian(self) -> SparsePauliOp:
        pauli_list = [("I" * self.num_nodes, -0.5 * sum(self.graph.edges()))]
        for edge in list(self.graph.edge_list()):
            paulis = ["I"] * self.num_nodes
            u, v = edge[0], edge[1]
            weight = self.graph.get_edge_data(u, v)

            paulis[u], paulis[v] = "Z", "Z"
            pauli_list.append(("".join(paulis)[::-1], 0.5 * weight))

        return SparsePauliOp.from_list(pauli_list)

    def get_ansatz(self):
        return self.ansatz

    def get_hamiltonian(self):
        return self.hamiltonian

    def get_initial_point(self, initialization_type):
        num_params = self.ansatz.num_parameters

        if initialization_type == "random":
            return 0.01 * 2 * np.pi * np.random.random(num_params)

        elif initialization_type == "small":
            return 0.01 * np.ones(num_params)

        elif initialization_type == "linear":
            params = np.zeros(num_params)
            half = num_params // 2

            beta_max = np.pi / (4 * self.p) if self.p > 0 else 0.1
            params[:half] = np.linspace(0.01, beta_max, half)

            gamma_max = np.pi / (2 * self.p) if self.p > 0 else 0.1
            params[half:] = np.linspace(0.01, gamma_max, half)

            return params

        elif initialization_type == "trotterized":
            params = np.zeros(num_params)
            half = num_params // 2

            beta_val = np.pi / (4 * self.p) if self.p > 0 else 0.1
            gamma_val = np.pi / (2 * self.p) if self.p > 0 else 0.1

            params[:half] = beta_val
            params[half:] = gamma_val

            return params

        else:
            raise ValueError(f"Unknown initialization type: {initialization_type}. "
                             f"Choose from: 'random', 'small', 'linear', 'trotterized'")

    def brute_force_max_cut(self):
        max_cut_value = 0
        best_partition = None

        for partition_bits in range(2 ** self.num_nodes):
            partition = [(partition_bits >> i) & 1 for i in range(self.num_nodes)]

            cut_value = 0
            for edge in self.graph.edge_list():
                u, v = edge[0], edge[1]
                weight = self.graph.get_edge_data(u, v)
                if partition[u] != partition[v]:
                    cut_value += weight

            if cut_value > max_cut_value:
                max_cut_value = cut_value
                best_partition = partition

        return max_cut_value, best_partition
