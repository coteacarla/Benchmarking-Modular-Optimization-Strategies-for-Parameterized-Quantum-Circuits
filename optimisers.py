import warnings
import numpy as np
from abc import ABC, abstractmethod

from scipy.optimize import minimize as scipy_minimize, OptimizeResult
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from gradients import (
    GradientStrategy,
    FiniteDifferenceGradient,
    SPSA_Gradient,
    PGPEGradient,
    PSRGradient,
    OCPGradient,
    PSRClassifierGradient,
    OCPClassifierGradient,
)


def update_early_stop_state(cost, best_loss, iterations_without_improvement, patience):
    if cost < best_loss:
        best_loss = cost
        iterations_without_improvement = 0
    else:
        iterations_without_improvement += 1

    should_stop = iterations_without_improvement >= patience
    return should_stop, best_loss, iterations_without_improvement


class Optimizer(ABC):
    @abstractmethod
    def minimize(self, func, x0):
        pass

class UpdateRule(ABC):
    @abstractmethod
    def update(self, params, grads, k_iter):
        pass

class SGD(UpdateRule):
    def __init__(self, learning_rate=0.1, decay=0.01):
        self.lr = learning_rate
        self.decay = decay

    def update(self, params, grads, k_iter):
        lr = self.lr / (1.0 + self.decay * k_iter) if self.decay > 0 else self.lr
        return params - lr * grads

class Adam(UpdateRule):
    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.m = None
        self.v = None

    def update(self, params, grads, k_iter):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** (k_iter + 1))
        v_hat = self.v / (1 - self.beta2 ** (k_iter + 1))

        lr = self.lr / (1.0 + self.decay * k_iter) if self.decay > 0 else self.lr
        return params - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(UpdateRule):
    def __init__(self, lr=0.05, beta=0.9, epsilon=1e-8, decay=0.01):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.decay = decay
        self.v = None

    def update(self, params, grads, k_iter):
        if self.v is None:
            self.v = np.zeros_like(params)

        self.v = self.beta * self.v + (1 - self.beta) * (grads ** 2)
        lr = self.lr / (1.0 + self.decay * k_iter) if self.decay > 0 else self.lr
        return params - lr * grads / (np.sqrt(self.v) + self.epsilon)


class ModularOptimizer(Optimizer):
    def __init__(self, gradient_strategy: GradientStrategy, update_rule: UpdateRule,
                 max_iter=100, early_stopping_patience=10, conv_tolerance=None):
        self.grad_strategy = gradient_strategy
        self.update_rule = update_rule
        self.max_iter = max_iter
        self.early_stopping_patience = early_stopping_patience
        self.conv_tolerance = conv_tolerance
        self.history = []
        self.parameter_history = []
        self.tracker = None

        self.ansatz = None
        self.hamiltonian = None

    def minimize(self, func, x0):
        self.history = []
        self.parameter_history = []
        params = np.array(x0, dtype=float)
        best_loss = float('inf')
        best_params = params.copy()
        iterations_without_improvement = 0
        func_evals = 0
        prev_cost = None
        small_change_streak = 0

        for k in range(self.max_iter):
            grads = self.grad_strategy.compute(params, func, k)
            last_cost = getattr(self.grad_strategy, 'last_cost', None)
            if last_cost is not None:
                cost = last_cost
                self.grad_strategy.last_cost = None
            else:
                cost = func(params)
                func_evals += 1
            self.history.append(cost)
            self.parameter_history.append(params.copy())

            previous_best = best_loss
            should_stop, best_loss, iterations_without_improvement = update_early_stop_state(
                cost, best_loss, iterations_without_improvement,
                self.early_stopping_patience,
            )
            if cost < previous_best:
                best_params = params.copy()

            if k % 5 == 0:
                print(f"  Iter {k}: Cost={cost:.4f}")

            if self.conv_tolerance is not None and prev_cost is not None:
                if abs(cost - prev_cost) < self.conv_tolerance:
                    small_change_streak += 1
                else:
                    small_change_streak = 0
            prev_cost = cost

            if self.conv_tolerance is not None:
                should_stop = small_change_streak >= self.early_stopping_patience

            if should_stop:
                if self.conv_tolerance is not None:
                    print(
                        f"  Early stopping after {self.early_stopping_patience} "
                        f"iterations with cost change below {self.conv_tolerance:.2e}."
                    )
                else:
                    print(f"  Early stopping after {self.early_stopping_patience} iterations without improvement.")
                if self.tracker:
                    self.tracker.record_iteration()
                break

            params = self.update_rule.update(params, grads, k)
            if self.tracker:
                self.tracker.record_iteration()

        res = OptimizeResult()
        res.x = best_params
        res.fun = best_loss
        res.success = True
        res.message = 'Optimization terminated successfully'
        res.nit = k + 1
        res.nfev = func_evals
        return res


class CobylaOptimizer(Optimizer):
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
        self.history = []
        self.parameter_history = []
        self.tracker = None

    def minimize(self, func, x0):
        self.history = []
        self.parameter_history = []

        def wrapped_func(x):
            val = func(x)
            self.history.append(val)
            self.parameter_history.append(np.array(x, copy=True))
            if self.tracker:
                self.tracker.record_iteration()
            return val

        res = scipy_minimize(wrapped_func, x0, method='COBYLA',
                             options={'maxiter': self.max_iter})
        return res


class NelderMeadOptimizer(Optimizer):
    def __init__(self, max_iter=100, initial_simplex_step=0.2):
        self.max_iter = max_iter
        self.initial_simplex_step = initial_simplex_step
        self.history = []
        self.parameter_history = []
        self.tracker = None

    def minimize(self, func, x0):
        self.history = []
        self.parameter_history = []
        x0 = np.array(x0, dtype=float)


        n = len(x0)
        simplex = np.tile(x0, (n + 1, 1))
        for i in range(n):
            simplex[i + 1, i] += self.initial_simplex_step

        def wrapped_func(x):
            val = func(x)
            self.history.append(val)
            self.parameter_history.append(np.array(x, copy=True))
            if self.tracker:
                self.tracker.record_iteration()
            return val

        res = scipy_minimize(wrapped_func, x0, method='Nelder-Mead',
                             options={'maxiter': self.max_iter,
                                      'initial_simplex': simplex})
        return res


class BayesianOptimizer(Optimizer):

    MAX_ITERATIONS = 1000

    def __init__(self, max_iter=20, n_initial_points=5, bounds=(-np.pi, np.pi)):
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        self.max_iter = min(max_iter, self.MAX_ITERATIONS)
        self.n_initial = n_initial_points
        self.bounds = bounds
        self.history = []
        self.best_history = []
        self.parameter_history = []
        self.tracker = None

        const_kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        matern_kernel = Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-3, 1e12))
        noise_kernel = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-12, 10.0))

        kernel = const_kernel * matern_kernel + noise_kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                           alpha=1e-6)

    def _expected_improvement(self, X, Y_sample, xi=0.01):
        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample_opt = np.min(Y_sample)

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def minimize(self, func, x0):
        self.history = []
        self.best_history = []
        self.parameter_history = []

        import operator
        budget = min(operator.index(self.max_iter), self.MAX_ITERATIONS)
        if budget < 1:
            raise ValueError("max_iter must be at least 1")
        evaluations = 0

        x0 = np.array(x0, dtype=float)
        dim = x0.size
        lower, upper = self.bounds

        X_sample = []
        Y_sample = []

        def _evaluate(x):
            nonlocal evaluations
            if evaluations >= budget:
                raise RuntimeError("BO objective evaluation budget exhausted")
            evaluations += 1
            val = float(func(x))
            best = val if not self.best_history else min(val, self.best_history[-1])
            self.history.append(val)
            self.best_history.append(best)
            self.parameter_history.append(np.array(x, copy=True))
            if self.tracker:
                self.tracker.record_iteration()
            return val

        y0 = _evaluate(x0)
        X_sample.append(x0.copy())
        Y_sample.append(y0)

        n_bootstrap = min(max(0, self.n_initial - 1), budget - 1)
        for _ in range(n_bootstrap):
            x_rand = np.random.uniform(lower, upper, size=dim)
            y_rand = _evaluate(x_rand)
            X_sample.append(x_rand)
            Y_sample.append(y_rand)

        remaining_iters = max(0, budget - len(Y_sample))
        n_candidates = max(200, 50 * dim)

        for k in range(remaining_iters):
            X_arr = np.array(X_sample)
            Y_arr = np.array(Y_sample)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                self.gp.fit(X_arr, Y_arr)

            X_candidates = np.random.uniform(lower, upper, size=(n_candidates, dim))
            ei = self._expected_improvement(X_candidates, Y_arr)
            x_next = X_candidates[int(np.argmax(ei))]

            y_next = _evaluate(x_next)
            X_sample.append(x_next)
            Y_sample.append(y_next)


        Y_arr = np.array(Y_sample)
        best_idx = int(np.argmin(Y_arr))

        res = OptimizeResult()
        res.x = np.array(X_sample[best_idx], dtype=float)
        res.fun = float(Y_arr[best_idx])
        res.success = True
        res.message = f'Objective evaluation budget exhausted ({evaluations}/{budget})'
        res.nit = remaining_iters
        res.nfev = len(Y_sample)
        return res

class FiniteDiffOptimizer(ModularOptimizer):
    def __init__(self, max_iter=30, epsilon=0.1, update_rule=None, early_stopping_patience=10, conv_tolerance=None):
        strategy = FiniteDifferenceGradient(epsilon=epsilon)
        updater = update_rule if update_rule is not None else SGD(learning_rate=0.05)
        super().__init__(strategy, updater, max_iter, early_stopping_patience=early_stopping_patience,
                         conv_tolerance=conv_tolerance)

class SPSAOptimizer(ModularOptimizer):
    def __init__(self, max_iter=30, update_rule=None, early_stopping_patience=10, conv_tolerance=None):
        strategy = SPSA_Gradient(c=0.1)
        updater = update_rule if update_rule is not None else SGD(learning_rate=0.05)
        super().__init__(strategy, updater, max_iter, early_stopping_patience=early_stopping_patience,
                         conv_tolerance=conv_tolerance)

class PGPEOptimizer(ModularOptimizer):
    def __init__(self, max_iter=30, sigma=0.1, n_samples=5, update_rule=None, early_stopping_patience=10, conv_tolerance=None):
        strategy = PGPEGradient(sigma=sigma, n_samples=n_samples)
        updater = update_rule if update_rule is not None else SGD(learning_rate=0.05)
        super().__init__(strategy, updater, max_iter, early_stopping_patience=early_stopping_patience,
                         conv_tolerance=conv_tolerance)

class GradientDescentPSR(ModularOptimizer):
    def __init__(self, estimator, ansatz, hamiltonian, max_iter=30, update_rule=None, early_stopping_patience=10, conv_tolerance=None):
        psr_strategy = PSRGradient(estimator, ansatz, hamiltonian)
        updater = update_rule if update_rule is not None else SGD(learning_rate=0.01)
        super().__init__(psr_strategy, updater, max_iter, early_stopping_patience=early_stopping_patience,
                         conv_tolerance=conv_tolerance)
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian

class GradientDescentOCP(ModularOptimizer):
    def __init__(
        self,
        estimator,
        ansatz,
        hamiltonian,
        max_iter=30,
        update_rule=None,
        regularization_epsilon=0.1,
        learning_rate=0.05,
        early_stopping_patience=10,
        conv_tolerance=None,
    ):
        ocp_strategy = OCPGradient(
            estimator,
            ansatz,
            hamiltonian,
            regularization_epsilon=regularization_epsilon,
        )
        updater = update_rule if update_rule is not None else SGD(learning_rate=learning_rate, decay=0.0)
        super().__init__(ocp_strategy, updater, max_iter, early_stopping_patience=early_stopping_patience,
                         conv_tolerance=conv_tolerance)
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian

class GradientDescentPSRClassifier(ModularOptimizer):
    def __init__(self, estimator, isa_circuit, isa_observable,
                 X_train, y_train, batch_size=10,
                 max_iter=30, update_rule=None, early_stopping_patience=10, conv_tolerance=None):
        psr_strategy = PSRClassifierGradient(
            estimator, isa_circuit, isa_observable,
            X_train, y_train, batch_size=batch_size,
        )
        updater = update_rule if update_rule is not None else SGD(learning_rate=0.05)
        super().__init__(psr_strategy, updater, max_iter, early_stopping_patience=early_stopping_patience,
                         conv_tolerance=conv_tolerance)

class GradientDescentOCPClassifier(ModularOptimizer):
    def __init__(
        self,
        estimator,
        isa_circuit,
        isa_observable,
        X_train,
        y_train,
        batch_size=10,
        max_iter=30,
        update_rule=None,
        regularization_epsilon=0.1,
        learning_rate=0.05,
        early_stopping_patience=10,
        conv_tolerance=None,
    ):
        ocp_strategy = OCPClassifierGradient(
            estimator, isa_circuit, isa_observable,
            X_train, y_train,
            batch_size=batch_size,
            regularization_epsilon=regularization_epsilon,
        )
        updater = update_rule if update_rule is not None else SGD(learning_rate=learning_rate, decay=0.0)
        super().__init__(ocp_strategy, updater, max_iter, early_stopping_patience=early_stopping_patience,
                         conv_tolerance=conv_tolerance)
