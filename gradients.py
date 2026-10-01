from abc import ABC, abstractmethod

import numpy as np

from base import squash_pubs

class GradientComputationError(Exception):
    pass


class GradientStrategy(ABC):
    @abstractmethod
    def compute(self, params, cost_func, k_iter):
        pass


class FiniteDifferenceGradient(GradientStrategy):
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon
        self.batch_eval = None

    def compute(self, params, cost_func, k_iter):
        n = len(params)
        grads = np.zeros_like(params)
        self.last_cost = None

        if self.batch_eval is not None:
            param_list = [params.copy()]
            for i in range(n):
                p_plus = params.copy(); p_plus[i] += self.epsilon
                p_minus = params.copy(); p_minus[i] -= self.epsilon
                param_list.append(p_plus)
                param_list.append(p_minus)
            costs = self.batch_eval(param_list)
            self.last_cost = costs[0]
            for i in range(n):
                grads[i] = (costs[1 + 2 * i] - costs[2 + 2 * i]) / (2 * self.epsilon)
        else:
            for i in range(n):
                p_plus = params.copy(); p_plus[i] += self.epsilon
                p_minus = params.copy(); p_minus[i] -= self.epsilon
                grads[i] = (cost_func(p_plus) - cost_func(p_minus)) / (2 * self.epsilon)

        return grads


class SPSA_Gradient(GradientStrategy):
    def __init__(self, c=0.1, gamma=0.101):
        self.c = c
        self.gamma = gamma
        self.batch_eval = None

    def compute(self, params, cost_func, k_iter):
        c_k = self.c / ((k_iter + 1) ** self.gamma)
        delta = 2 * np.random.randint(0, 2, size=params.shape) - 1
        self.last_cost = None

        if self.batch_eval is not None:
            costs = self.batch_eval([params, params + c_k * delta, params - c_k * delta])
            self.last_cost = costs[0]
            y_plus, y_minus = costs[1], costs[2]
        else:
            y_plus = cost_func(params + c_k * delta)
            y_minus = cost_func(params - c_k * delta)

        ghat = (y_plus - y_minus) / (2 * c_k * delta)
        return ghat


class PGPEGradient(GradientStrategy):
    def __init__(self, sigma=0.1, n_samples=5):
        self.sigma = sigma
        self.n_samples = n_samples
        self.batch_eval = None

    def compute(self, params, cost_func, k_iter):
        grads = np.zeros_like(params)
        self.last_cost = None
        perturbations = [
            np.random.normal(0, self.sigma, size=params.shape)
            for _ in range(self.n_samples)
        ]

        if self.batch_eval is not None:
            param_list = [params.copy()]
            for eps in perturbations:
                param_list.append(params + eps)
                param_list.append(params - eps)
            costs = self.batch_eval(param_list)
            self.last_cost = costs[0]
            for idx, eps in enumerate(perturbations):
                f_plus  = costs[1 + 2 * idx]
                f_minus = costs[2 + 2 * idx]
                grads += (f_plus - f_minus) * eps
        else:
            for eps in perturbations:
                f_plus  = cost_func(params + eps)
                f_minus = cost_func(params - eps)
                grads += (f_plus - f_minus) * eps

        grads /= (2 * self.n_samples * self.sigma ** 2)
        return grads


class PSRGradient(GradientStrategy):

    def __init__(self, estimator, ansatz, hamiltonian, shift=np.pi / 2):
        from qiskit.circuit import ParameterVector

        if not np.isfinite(shift) or abs(np.sin(shift)) < 1e-12:
            raise ValueError("The parameter shift must have nonzero sine.")
        self.estimator = estimator
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.shift = shift
        self.gradient_factor = 1.0 / (2 * np.sin(shift))
        self.tracker = None
        self.cost_transform = None
        self.parameters = tuple(ansatz.parameters)
        self.observable = (hamiltonian.apply_layout(ansatz.layout)
                           if ansatz.layout is not None else hamiltonian)

        circuit = ansatz.copy()
        supported = {"rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "rzx"}
        for _ in range(20):
            unsupported = {inst.operation.name for inst in circuit.data
                           if inst.operation.is_parameterized()
                           and inst.operation.name not in supported}
            if not unsupported:
                break
            circuit = circuit.decompose(gates_to_decompose=list(unsupported))
        else:
            raise ValueError("Cannot decompose ansatz into supported parameterized rotations.")


        circuit.global_phase = 0
        sites = [(i, inst.operation.params[0]) for i, inst in enumerate(circuit.data)
                 if inst.operation.is_parameterized()]
        gate_parameters = ParameterVector("_psr_gate", len(sites))
        self.expressions = []
        self.jacobian = np.zeros((len(sites), len(self.parameters)))
        for l, (i, expression) in enumerate(sites):
            self.expressions.append(expression)
            for j, parameter in enumerate(self.parameters):
                derivative = expression.gradient(parameter)
                if getattr(derivative, "parameters", set()):
                    raise ValueError("Gate-wise PSR currently requires affine gate angles.")
                self.jacobian[l, j] = float(derivative)
            operation = circuit.data[i].operation.to_mutable()
            operation.params = [gate_parameters[l]]
            circuit.data[i] = circuit.data[i].replace(operation=operation)
        self.circuit = circuit
        self.gate_parameters = tuple(gate_parameters)

    def _plans(self, curvature):
        from collections import defaultdict

        zero = (0,) * len(self.gate_parameters)
        gradients, hessians = [], []
        for j in range(len(self.parameters)):
            terms = []
            for l in np.flatnonzero(self.jacobian[:, j]):
                for sign in (1, -1):
                    offset = list(zero)
                    offset[l] = sign
                    terms.append((tuple(offset), sign * self.gradient_factor * self.jacobian[l, j]))
            gradients.append(dict(terms))
            if curvature:
                weights = defaultdict(float)

                for left, a in terms:
                    for right, b in terms:
                        offset = tuple(x + y for x, y in zip(left, right))
                        weights[offset] += a * b
                hessians.append(dict(weights))
        return zero, gradients, hessians

    def _evaluate(self, params, curvature=False):
        params = np.asarray(params, dtype=float)
        if params.shape != (len(self.parameters),):
            raise ValueError("Parameter vector has the wrong shape.")
        bindings = dict(zip(self.parameters, params))
        angles = np.array([float(expr.bind({p: bindings[p] for p in expr.parameters}))
                           for expr in self.expressions])
        zero, gradient_plans, hessian_plans = self._plans(curvature)
        offsets = dict.fromkeys([zero] + [offset for plan in gradient_plans + hessian_plans
                                        for offset, weight in plan.items() if weight != 0])
        pubs = []
        for offset in offsets:
            values = angles + self.shift * np.asarray(offset)
            pubs.append((self.circuit, self.observable, dict(zip(self.gate_parameters, values))))

        pubs = [(c, o, [v[p] for p in c.parameters]) for c, o, v in pubs]
        try:
            results = self.estimator.run(pubs).result()
            if self.tracker:
                self.tracker.add_evaluations(len(pubs))
            values = {}
            for offset, result in zip(offsets, results):
                value = float(np.asarray(result.data.evs).item())
                values[offset] = float(self.cost_transform(value)) if self.cost_transform else value
            self.last_cost = values[zero]
            def evaluate(plans):
                return np.array([sum(weight * values[offset] for offset, weight in plan.items()
                                     if weight != 0) for plan in plans])
            return evaluate(gradient_plans), evaluate(hessian_plans)
        except Exception as e:
            raise GradientComputationError(f"Gate-wise shift computation failed: {e}") from e

    def compute(self, params, cost_func, k_iter):
        return self._evaluate(params)[0]


class OCPGradient(PSRGradient):

    def __init__(self, estimator, ansatz, hamiltonian, regularization_epsilon=0.1):
        super().__init__(estimator, ansatz, hamiltonian)
        if regularization_epsilon <= 0:
            raise ValueError("Curvature regularization must be positive.")
        self.regularization_epsilon = regularization_epsilon

    def compute(self, params, cost_func, k_iter):
        gradient, hessian = self._evaluate(params, curvature=True)
        return gradient / (np.abs(hessian) + self.regularization_epsilon)


class PSRClassifierGradient(GradientStrategy):
    def __init__(self, estimator, isa_circuit, isa_observable,
                 X_train, y_train, batch_size=10, shift=np.pi / 2):
        self.estimator = estimator
        self.isa_circuit = isa_circuit
        self.isa_observable = isa_observable
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.shift = shift
        self.gradient_factor = 1.0 / (2 * np.sin(self.shift))
        self.tracker = None

    def compute(self, params, cost_func, k_iter):
        n_theta = len(params)

        indices = np.random.choice(
            len(self.X_train),
            size=min(self.batch_size, len(self.X_train)),
            replace=False,
        )
        X_batch = self.X_train[indices]
        y_batch = self.y_train[indices]
        M = len(X_batch)

        pubs = []

        for x in X_batch:
            full = np.concatenate([x, params])
            pubs.append((self.isa_circuit, self.isa_observable, full.tolist()))

        for j in range(n_theta):
            p_plus = params.copy()
            p_plus[j] += self.shift
            p_minus = params.copy()
            p_minus[j] -= self.shift
            for x in X_batch:
                pubs.append((self.isa_circuit, self.isa_observable,
                             np.concatenate([x, p_plus]).tolist()))
                pubs.append((self.isa_circuit, self.isa_observable,
                             np.concatenate([x, p_minus]).tolist()))

        try:
            results = self.estimator.run([squash_pubs(pubs)]).result()
            if self.tracker:
                self.tracker.add_evaluations(len(pubs))
        except Exception as e:
            raise GradientComputationError(f"PSR classifier gradient computation failed: {e}") from e

        evs_all = np.asarray(results[0].data.evs)
        ev_current = evs_all[:M]
        p_pred = (1.0 + ev_current) / 2.0
        self.last_cost = float(np.mean((p_pred - y_batch) ** 2))

        grads = np.zeros(n_theta)
        idx = M
        for j in range(n_theta):
            grad_j = 0.0
            for s in range(M):
                ev_plus = float(evs_all[idx])
                ev_minus = float(evs_all[idx + 1])
                idx += 2

                dZ_dtheta = self.gradient_factor * (ev_plus - ev_minus)
                grad_j += (p_pred[s] - y_batch[s]) * dZ_dtheta

            grads[j] = grad_j / M

        return grads


class OCPClassifierGradient(GradientStrategy):
    def __init__(
        self,
        estimator,
        isa_circuit,
        isa_observable,
        X_train,
        y_train,
        batch_size=10,
        regularization_epsilon=0.1,
    ):
        self.estimator = estimator
        self.isa_circuit = isa_circuit
        self.isa_observable = isa_observable
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.regularization_epsilon = regularization_epsilon
        self.tracker = None

    def compute(self, params, cost_func, k_iter):
        params = np.asarray(params, dtype=float)
        n_theta = len(params)

        indices = np.random.choice(
            len(self.X_train),
            size=min(self.batch_size, len(self.X_train)),
            replace=False,
        )
        X_batch = self.X_train[indices]
        y_batch = self.y_train[indices]
        M = len(X_batch)

        pubs = []

        for x in X_batch:
            full = np.concatenate([x, params])
            pubs.append((self.isa_circuit, self.isa_observable, full.tolist()))


        for j in range(n_theta):
            p_pp = params.copy(); p_pp[j] += np.pi / 2
            p_pm = params.copy(); p_pm[j] -= np.pi / 2
            p_hp = params.copy(); p_hp[j] += np.pi
            p_hm = params.copy(); p_hm[j] -= np.pi
            for x in X_batch:
                pubs.append((self.isa_circuit, self.isa_observable,
                             np.concatenate([x, p_pp]).tolist()))
                pubs.append((self.isa_circuit, self.isa_observable,
                             np.concatenate([x, p_pm]).tolist()))
                pubs.append((self.isa_circuit, self.isa_observable,
                             np.concatenate([x, p_hp]).tolist()))
                pubs.append((self.isa_circuit, self.isa_observable,
                             np.concatenate([x, p_hm]).tolist()))

        try:
            results = self.estimator.run([squash_pubs(pubs)]).result()
            if self.tracker:
                self.tracker.add_evaluations(len(pubs))
        except Exception as e:
            raise GradientComputationError(f"OCP classifier gradient computation failed: {e}") from e

        evs_all = np.asarray(results[0].data.evs)
        ev_current = evs_all[:M]
        p_pred = (1.0 + ev_current) / 2.0
        self.last_cost = float(np.mean((p_pred - y_batch) ** 2))

        grads = np.zeros(n_theta)
        hess_diag = np.zeros(n_theta)

        idx = M
        for j in range(n_theta):
            grad_j = 0.0
            hess_j = 0.0
            for s in range(M):
                ev_pp = float(evs_all[idx])
                ev_pm = float(evs_all[idx + 1])
                ev_hp = float(evs_all[idx + 2])
                ev_hm = float(evs_all[idx + 3])
                ev_0  = ev_current[s]
                idx += 4

                dZ_dtheta = 0.5 * (ev_pp - ev_pm)
                grad_j += (p_pred[s] - y_batch[s]) * dZ_dtheta
                hess_j += abs((ev_hp + ev_hm - 2.0 * ev_0) / 4.0)

            grads[j] = grad_j / M
            hess_diag[j] = hess_j / M

        metric = hess_diag + self.regularization_epsilon
        return grads / metric
