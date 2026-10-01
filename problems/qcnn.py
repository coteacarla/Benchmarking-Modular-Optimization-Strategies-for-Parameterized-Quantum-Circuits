import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from sklearn.model_selection import train_test_split

from base import QuantumProblem


def _qcnn_conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def _qcnn_pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


def _qcnn_conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(_qcnn_conv_circuit(params[param_index:(param_index + 3)]), [q1, q2])
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(_qcnn_conv_circuit(params[param_index:(param_index + 3)]), [q1, q2])
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def _qcnn_pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(_qcnn_pool_circuit(params[param_index:(param_index + 3)]), [source, sink])
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


class QCNNClassifierProblem(QuantumProblem):
    def __init__(self, num_images_per_class: int = 100,
                 test_size: float = 0.3, seed: int = 42):
        self.n_qubits = 8
        self.seed = seed
        self.num_images_per_class = num_images_per_class

        self.X_train, self.X_test, self.y_train, self.y_test = \
            self._prepare_data(num_images_per_class, test_size)

        self.x_params = ParameterVector("a", self.n_qubits)
        self._ansatz = self._build_ansatz()
        self._circuit = self._build_full_circuit()

        self._observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    def _prepare_data(self, num_images_per_class, test_size):
        from torchvision import datasets as tv_datasets
        from skimage.transform import resize
        from skimage.filters import threshold_otsu

        mnist = tv_datasets.MNIST(root="./data", train=True, download=True)
        x_all = mnist.data.numpy().astype(np.float32)
        y_all = mnist.targets.numpy()

        digit_0 = x_all[y_all == 0][:num_images_per_class]
        digit_1 = x_all[y_all == 1][:num_images_per_class]

        def preprocess(image):
            image = image / 255.0
            resized = resize(image, (8, 8), order=3,
                             anti_aliasing=True, mode="reflect")
            thresh = threshold_otsu(resized)
            binary = (resized > thresh).astype(int)
            return binary * (np.pi / 2)

        images, labels = [], []
        for img in digit_0:
            images.append(preprocess(img))
            labels.append(0)
        for img in digit_1:
            images.append(preprocess(img))
            labels.append(1)

        images = np.array(images)
        labels = np.array(labels)

        X = images.reshape((-1, 8, 8)).mean(axis=2)

        return train_test_split(X, labels,
                                test_size=test_size,
                                random_state=self.seed)

    def _build_feature_map(self):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.h(i)
            qc.p(2.0 * self.x_params[i], i)
        return qc

    def _build_ansatz(self):
        ansatz = QuantumCircuit(self.n_qubits, name="QCNN_Ansatz")

        ansatz.compose(_qcnn_conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(_qcnn_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"),
                       list(range(8)), inplace=True)

        ansatz.compose(_qcnn_conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(_qcnn_pool_layer([0, 1], [2, 3], "p2"),
                       list(range(4, 8)), inplace=True)

        ansatz.compose(_qcnn_conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(_qcnn_pool_layer([0], [1], "p3"),
                       list(range(6, 8)), inplace=True)
        return ansatz

    def _build_full_circuit(self):
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self._build_feature_map(), range(self.n_qubits), inplace=True)
        circuit.compose(self._ansatz, range(self.n_qubits), inplace=True)
        return circuit.decompose()

    def get_ansatz(self):
        return self._circuit

    def get_hamiltonian(self):
        return self._observable

    def get_initial_point(self, initialization_type: str = "small"):
        n = len(list(self._ansatz.parameters))
        if initialization_type == "small":
            return 0.01 * np.ones(n)
        elif initialization_type == "random":
            return 2 * np.pi * np.random.random(n)
        else:
            raise ValueError(f"Unknown type: {initialization_type}")

    def input_params(self):
        return list(self.x_params)

    def weight_params(self):
        return sorted(self._ansatz.parameters, key=lambda p: p.name)

    def dataset_summary(self):
        print(f"  Dataset        : MNIST (digits 0 vs 1)")
        print(f"  Images/class   : {self.num_images_per_class}")
        print(f"  Train samples  : {len(self.X_train)}")
        print(f"  Test  samples  : {len(self.X_test)}")
        print(f"  Features       : {self.n_qubits}")
        print(f"  Circuit qubits : {self.n_qubits}")
        print(f"  Trainable params: {len(self.weight_params())}")
        class_counts = np.bincount(self.y_train)
        print(f"  Train class balance: {dict(enumerate(class_counts))}")
