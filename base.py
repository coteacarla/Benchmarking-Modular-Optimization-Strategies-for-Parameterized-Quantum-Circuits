import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


REPORT_EPOCHS = (50, 100)

class QuantumProblem(ABC):
    @abstractmethod
    def get_ansatz(self) -> QuantumCircuit:
        pass

    @abstractmethod
    def get_hamiltonian(self) -> SparsePauliOp:
        pass

    @abstractmethod
    def get_initial_point(self) -> np.ndarray:
        pass

class ConvergenceTracker:
    def __init__(self):
        self.circuit_evaluations: int = 0
        self.eval_history: List[int] = []
        self.time_history: List[float] = []
        self.start_time: Optional[float] = None
        self.wall_clock_time: float = 0.0

    def start(self):
        self.circuit_evaluations = 0
        self.eval_history = []
        self.time_history = []
        self.start_time = time.time()
        self.wall_clock_time = 0.0

    def stop(self):
        if self.start_time is not None:
            self.wall_clock_time = time.time() - self.start_time

    def add_evaluations(self, n: int = 1):
        self.circuit_evaluations += n

    def record_iteration(self):
        self.eval_history.append(self.circuit_evaluations)
        if self.start_time is not None:
            self.time_history.append(time.time() - self.start_time)


    def wrap_cost_function(self, func, evals_per_call: int = 1):
        tracker = self

        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            tracker.add_evaluations(evals_per_call)
            return result

        return wrapped


    @staticmethod
    def find_first_small_change_iteration(history, tolerance=1e-4):
        return next((i for i in range(1, len(history))
                     if abs(history[i] - history[i - 1]) < tolerance), None)

    @staticmethod
    def find_stability_iteration(history, tolerance=1e-4, patience=10):
        if patience < 1:
            raise ValueError("Stability patience must be at least 1")
        streak = 0
        for i in range(1, len(history)):
            streak = streak + 1 if abs(history[i] - history[i - 1]) < tolerance else 0
            if streak >= patience:
                return i
        return None

    @staticmethod
    def find_convergence_iteration(
        history: List[float],
        tolerance: float = 1e-6,
        mode: str = "cost_diff",
        target_value: Optional[float] = None,
        patience: int = 10,
    ) -> Optional[int]:
        if patience < 1:
            raise ValueError("Convergence patience must be at least 1")
        if mode == "cost_diff":
            return ConvergenceTracker.find_stability_iteration(history, tolerance, patience)
        elif mode == "threshold" and target_value is not None:
            for i, val in enumerate(history):
                if val <= target_value + tolerance:
                    return i
        return None


def squash_pubs(pubs):
    if not pubs:
        raise ValueError("pubs list is empty")
    circuit, observable = pubs[0][0], pubs[0][1]
    param_matrix = np.array([np.asarray(p[2]) for p in pubs])
    return (circuit, observable, param_matrix)


def run_with_tracking(
    optimizer,
    cost_func,
    initial_point,
    evals_per_call: int = 1,
    conv_tolerance: float = 1e-6,
    conv_mode: str = "cost_diff",
    conv_target: Optional[float] = None,
):
    tracker = ConvergenceTracker()
    wrapped = tracker.wrap_cost_function(cost_func, evals_per_call=evals_per_call)


    optimizer.tracker = tracker
    gs = getattr(optimizer, "grad_strategy", None)
    if gs is not None and hasattr(gs, "tracker"):
        gs.tracker = tracker
    _orig_batch = getattr(gs, "batch_eval", None)
    if _orig_batch is not None:
        def _tracked_batch(params_list, _orig=_orig_batch):
            results = _orig(params_list)
            tracker.add_evaluations(len(params_list) * evals_per_call)
            return results
        gs.batch_eval = _tracked_batch

    tracker.start()
    try:
        res_obj = optimizer.minimize(wrapped, initial_point)
    finally:
        tracker.stop()
        if _orig_batch is not None:
            gs.batch_eval = _orig_batch

    conv_iter = ConvergenceTracker.find_convergence_iteration(
        optimizer.history, tolerance=conv_tolerance, mode=conv_mode,
        target_value=conv_target,
        patience=getattr(optimizer, "early_stopping_patience", 10),
    )
    conv_evals = (
        tracker.eval_history[conv_iter]
        if conv_iter is not None and conv_iter < len(tracker.eval_history)
        else None
    )
    conv_time = (
        tracker.time_history[conv_iter]
        if conv_iter is not None and conv_iter < len(tracker.time_history)
        else None
    )
    return res_obj, tracker, conv_iter, conv_evals, conv_time


class OptimizationResult:
    def __init__(self, optimizer_name: str, problem_name: str, final_cost: float,
                 history: List[float], noise_level: str = "None",
                 wall_clock_time: float = 0.0,
                 circuit_evaluations: int = 0,
                 eval_history: List[int] = None,
                 time_history: List[float] = None,
                 convergence_iteration: Optional[int] = None,
                 convergence_evals: Optional[int] = None,
                 convergence_time: Optional[float] = None):
        self.optimizer_name = optimizer_name
        self.problem_name = problem_name
        self.final_cost = final_cost
        self.history = history
        self.noise_level = noise_level

        self.wall_clock_time = wall_clock_time
        self.circuit_evaluations = circuit_evaluations
        self.eval_history = eval_history or []
        self.time_history = time_history or []
        self.convergence_iteration = convergence_iteration
        self.convergence_evals = convergence_evals
        self.convergence_time = convergence_time


class ClassificationResult:
    def __init__(self, optimizer_name: str, problem_name: str,
                 final_loss: float, loss_history: List[float],
                 train_acc: float, test_acc: float,
                 train_acc_history: List[float] = None,
                 test_acc_history: List[float] = None,
                 noise_level: str = "None",
                 wall_clock_time: float = 0.0,
                 circuit_evaluations: int = 0,
                 eval_history: List[int] = None,
                 time_history: List[float] = None,
                 convergence_iteration: Optional[int] = None,
                 convergence_evals: Optional[int] = None,
                 convergence_time: Optional[float] = None):
        self.optimizer_name = optimizer_name
        self.problem_name = problem_name
        self.final_loss = final_loss
        self.loss_history = loss_history
        self.train_acc = train_acc
        self.test_acc = test_acc
        self.train_acc_history = train_acc_history or []
        self.test_acc_history = test_acc_history or []
        self.noise_level = noise_level

        self.wall_clock_time = wall_clock_time
        self.circuit_evaluations = circuit_evaluations
        self.eval_history = eval_history or []
        self.time_history = time_history or []
        self.convergence_iteration = convergence_iteration
        self.convergence_evals = convergence_evals
        self.convergence_time = convergence_time


def checkpoint_result(result, steps, tolerance=1e-4, patience=10):
    from copy import copy

    snapshot = copy(result)
    if hasattr(snapshot, "checkpoints"):
        del snapshot.checkpoints
    history_name = "loss_history" if hasattr(result, "loss_history") else "history"
    history = list(getattr(result, history_name)[:steps])
    if not history:
        raise ValueError("Cannot report an empty optimization history")
    setattr(snapshot, history_name, history)
    snapshot.requested_steps = steps
    snapshot.recorded_steps = len(history)
    snapshot.eval_history = list(result.eval_history[:len(history)])
    snapshot.time_history = list(result.time_history[:len(history)])
    snapshot.circuit_evaluations = snapshot.eval_history[-1] if snapshot.eval_history else 0
    snapshot.wall_clock_time = snapshot.time_history[-1] if snapshot.time_history else 0.0
    value_name = "final_loss" if history_name == "loss_history" else "final_cost"
    setattr(snapshot, value_name, float(min(history)))
    ci = ConvergenceTracker.find_convergence_iteration(history, tolerance=tolerance, patience=patience)
    snapshot.convergence_iteration = ci
    snapshot.convergence_evals = snapshot.eval_history[ci] if ci is not None and ci < len(snapshot.eval_history) else None
    snapshot.convergence_time = snapshot.time_history[ci] if ci is not None and ci < len(snapshot.time_history) else None
    attach_trajectory_diagnostics(snapshot, tolerance, patience)
    if history_name == "loss_history":

        snapshot.train_acc = float("nan")
        snapshot.test_acc = float("nan")
        snapshot.train_acc_history = []
        snapshot.test_acc_history = []
    return snapshot


def attach_trajectory_diagnostics(result, tolerance=1e-4, patience=10):
    history = result.loss_history if hasattr(result, "loss_history") else result.history
    for name, index in (
        ("first_small_change", ConvergenceTracker.find_first_small_change_iteration(history, tolerance)),
        ("stability", ConvergenceTracker.find_stability_iteration(history, tolerance, patience)),
    ):
        setattr(result, name + "_iteration", index)
        for suffix, values in (("evals", result.eval_history), ("time", result.time_history)):
            setattr(result, name + "_" + suffix,
                    values[index] if index is not None and index < len(values) else None)
    return result


def attach_checkpoints(result, optimizer, tolerance=1e-4, evaluate=None):
    attach_trajectory_diagnostics(result, tolerance,
                                  getattr(optimizer, "early_stopping_patience", 10))
    result.recorded_steps = len(optimizer.history)
    result.checkpoints = {}
    for steps in REPORT_EPOCHS:
        snapshot = checkpoint_result(
            result, steps, tolerance,
            patience=getattr(optimizer, "early_stopping_patience", 10))
        if evaluate is not None:
            index = int(np.argmin(optimizer.history[:steps]))
            _, snapshot.train_acc, snapshot.test_acc = evaluate(optimizer.parameter_history[index])
        result.checkpoints[steps] = snapshot
    return result


def checkpoint_groups(results):
    for steps in REPORT_EPOCHS:
        yield steps, [result.checkpoints[steps] for result in results]

    yield "full", list(results)
