import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from base import QuantumProblem


class IrisClassifierProblem(QuantumProblem):
    def __init__(self, n_layers: int = 2, binary_mode: str = "0v1",
                 test_size: float = 0.2, seed: int = 42):
        self.n_qubits = 4
        self.n_layers = n_layers
        self.binary_mode = binary_mode
        self.seed = seed

        self.X_train, self.X_test, self.y_train, self.y_test = \
            self._prepare_data(test_size)

        self.theta = ParameterVector("θ", self._num_trainable_params())
        self.x_params = ParameterVector("x", self.n_qubits)
        self._circuit = self._build_circuit()

        z_label = ["I"] * self.n_qubits
        z_label[0] = "Z"
        self._observable = SparsePauliOp.from_list([("".join(z_label)[::-1], 1.0)])

    def _prepare_data(self, test_size):
        iris = load_iris()
        X, y = iris.data, iris.target

        if self.binary_mode == "0v1":
            mask = y <= 1
        elif self.binary_mode == "0vRest":
            y = (y > 0).astype(int)
            mask = np.ones(len(y), dtype=bool)
        else:
            raise ValueError(f"Unknown binary_mode: {self.binary_mode}")

        X, y = X[mask], y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        self.scaler = MinMaxScaler(feature_range=(0, np.pi), clip=False)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def _num_trainable_params(self):
        return 2 * self.n_qubits * self.n_layers

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)

        for i in range(self.n_qubits):
            qc.rx(self.x_params[i], i)

        qc.barrier()

        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(self.theta[idx], q)
                qc.rz(self.theta[idx + 1], q)
                idx += 2
            for q in range(self.n_qubits):
                qc.cx(q, (q + 1) % self.n_qubits)

        return qc

    def get_ansatz(self):
        return self._circuit

    def get_hamiltonian(self):
        return self._observable

    def get_initial_point(self, initialization_type: str = "small"):
        n = self._num_trainable_params()
        if initialization_type == "small":
            return 0.01 * np.ones(n)
        elif initialization_type == "random":
            return 2 * np.pi * np.random.random(n)
        else:
            raise ValueError(f"Unknown type: {initialization_type}")

    def bind_features(self, x_sample: np.ndarray) -> dict:
        return dict(zip(self.x_params, x_sample))

    def num_classes(self):
        return 2

    def dataset_summary(self):
        print(f"  Mode           : {self.binary_mode}")
        print(f"  Train samples  : {len(self.X_train)}")
        print(f"  Test  samples  : {len(self.X_test)}")
        print(f"  Features       : {self.n_qubits}")
        print(f"  Ansatz layers  : {self.n_layers}")
        print(f"  Trainable θ    : {self._num_trainable_params()}")
        class_counts = np.bincount(self.y_train)
        print(f"  Train class balance: {dict(enumerate(class_counts))}")
