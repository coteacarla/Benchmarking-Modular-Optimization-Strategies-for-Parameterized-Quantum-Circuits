import contextlib
import random

import numpy as np
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService, Session

from base import (
    ClassificationResult,
    attach_checkpoints,
    checkpoint_groups,
    run_with_tracking,
    squash_pubs,
)
from optimisers import (
    Adam,
    BayesianOptimizer,
    CobylaOptimizer,
    FiniteDiffOptimizer,
    GradientDescentPSRClassifier,
    GradientDescentOCPClassifier,
    NelderMeadOptimizer,
    PGPEOptimizer,
    RMSprop,
    SGD,
    SPSAOptimizer,
)
from problems import IrisClassifierProblem
from reporter import (
    plot_convergence_metrics,
    plot_iris_results,
    print_category,
    print_convergence_summary,
    print_iris_results_summary,
    print_iris_setup,
    print_optimizer_start,
    save_convergence_report,
    save_checkpoint_table,
)


simulation = True
BACKEND = "ibm_aachen"

SEED = 42
BINARY_MODE = "0v1"
N_LAYERS = 2
INIT_TYPE = "random"
EPOCHS = 100
MAX_ITER = EPOCHS
BATCH_SIZE = 16
SHOTS = 256
EARLY_STOPPING = 10
CONV_TOLERANCE = 1e-4
grad_free_iters = 100

def _make_rules(lr):
    return {
        'SGD': SGD(learning_rate=lr, decay=0.05),
        'Adam': Adam(lr=lr, decay=0.05),
        'RMSprop': RMSprop(lr=lr, decay=0.05),
    }


def all_optimizers_experiment(
    train_optimizer, initial_point, grad_free_iters,
    estimator, isa_circuit, isa_observable, X_train, y_train,
    save_prefix, title_suffix
):
    results = []

    print_category(1, "GRADIENT-FREE OPTIMIZERS")
    results.append(train_optimizer("COBYLA", CobylaOptimizer(max_iter=grad_free_iters), initial_point.copy()))
    results.append(train_optimizer("Nelder-Mead", NelderMeadOptimizer(max_iter=grad_free_iters), initial_point.copy()))
    results.append(train_optimizer("Bayesian", BayesianOptimizer(max_iter=MAX_ITER, n_initial_points=5), initial_point.copy()))

    print_category(2, "GRADIENT APPROXIMATIONS (All Update Rules)")
    approx_classes = {
        'FiniteDiff': FiniteDiffOptimizer,
        'SPSA': SPSAOptimizer,
        'PGPE': PGPEOptimizer,
    }
    for opt_name, optimizer_cls in approx_classes.items():
        for rule_name, rule in _make_rules(lr=0.05).items():
            np.random.seed(SEED)
            random.seed(SEED)
            results.append(train_optimizer(
                f"{opt_name}-{rule_name}",
                optimizer_cls(max_iter=EPOCHS, update_rule=rule,
                         early_stopping_patience=EARLY_STOPPING,
                         conv_tolerance=CONV_TOLERANCE),
                initial_point.copy()))

    print_category(3, "EXACT GRADIENTS — Parameter Shift Rule (Classifier)")
    for rule_name, rule in _make_rules(lr=0.05).items():
        np.random.seed(SEED)
        random.seed(SEED)
        results.append(train_optimizer(
            f"PSR-{rule_name}",
            GradientDescentPSRClassifier(
                estimator, isa_circuit, isa_observable,
                X_train, y_train,
                batch_size=BATCH_SIZE, max_iter=EPOCHS,
                update_rule=rule, early_stopping_patience=EARLY_STOPPING,
                conv_tolerance=CONV_TOLERANCE),
            initial_point.copy()))
        rule = _make_rules(lr=0.05)[rule_name]
        results.append(train_optimizer(
            f"OCP-{rule_name}",
            GradientDescentOCPClassifier(
                estimator, isa_circuit, isa_observable,
                X_train, y_train,
                batch_size=BATCH_SIZE, max_iter=EPOCHS,
                update_rule=rule, regularization_epsilon=1e-3,
                early_stopping_patience=EARLY_STOPPING,
                conv_tolerance=CONV_TOLERANCE),
            initial_point.copy()))

    for steps, checkpoint_results in checkpoint_groups(results):
        checkpoint_prefix = f"{save_prefix}_{steps}epochs_{SHOTS}shots"
        print(f"\nReporting checkpoint: {steps} recorded steps")
        best = print_iris_results_summary(checkpoint_results)
        plot_iris_results(checkpoint_results, best, n_epochs=steps, initialization=INIT_TYPE, shots=SHOTS, save_prefix=checkpoint_prefix)
        print_convergence_summary(checkpoint_results, tolerance_info=f"loss_diff < {CONV_TOLERANCE}")
        plot_convergence_metrics(checkpoint_results, save_prefix=checkpoint_prefix, title_suffix=title_suffix)
        save_convergence_report(checkpoint_results, save_prefix=checkpoint_prefix, tolerance_info=f"loss_diff < {CONV_TOLERANCE}")
        save_checkpoint_table(checkpoint_results, checkpoint_prefix, steps, shots=SHOTS)


def run_iris_experiment():
    np.random.seed(SEED)
    random.seed(SEED)

    problem = IrisClassifierProblem(n_layers=N_LAYERS, binary_mode=BINARY_MODE, seed=SEED)

    if simulation:
        backend = AerSimulator(seed_simulator=SEED)
        session_ctx = contextlib.nullcontext()
        backend_label = "simulator"
        save_prefix, title_suffix = "iris", " — Iris Classifier"
    else:
        service = QiskitRuntimeService(channel='ibm_quantum_platform')
        backend = service.backend(BACKEND)
        session_ctx = Session(backend=backend)
        backend_label = "hardware"
        save_prefix, title_suffix = "iris_hw", " — Iris Classifier (Hardware)"

    print_iris_setup(backend.name, problem)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=SEED)
    isa_circuit = pm.run(problem.get_ansatz())
    isa_observable = problem.get_hamiltonian().apply_layout(isa_circuit.layout)

    X_train, y_train = problem.X_train, problem.y_train
    X_test, y_test = problem.X_test, problem.y_test
    initial_point = problem.get_initial_point(initialization_type=INIT_TYPE)

    with session_ctx as active_session:
        estimator = Estimator(mode=backend if simulation else active_session)
        estimator.options.default_shots = SHOTS
        if not simulation:
            estimator.options.resilience.measure_mitigation = True
            estimator.options.resilience.zne_mitigation = False
            estimator.options.resilience.pec_mitigation = False
            estimator.options.twirling.enable_gates = False
            estimator.options.twirling.enable_measure = True
            estimator.options.dynamical_decoupling.enable = False

        def cost_func(params):
            indices = np.random.choice(len(X_train), size=min(BATCH_SIZE, len(X_train)), replace=False)
            X_batch, y_batch = X_train[indices], y_train[indices]
            pubs = [(isa_circuit, isa_observable, np.concatenate([x, params]).tolist()) for x in X_batch]
            if simulation:
                res = estimator.run(pubs).result()
                evs = [res[i].data.evs for i in range(len(X_batch))]
            else:
                res = estimator.run([squash_pubs(pubs)]).result()
                evs = res[0].data.evs
            losses = [((1.0 + float(evs[i])) / 2.0 - y_batch[i]) ** 2
                      for i in range(len(X_batch))]
            return float(np.mean(losses))

        def train_and_evaluate(params):
            pubs_tr = [(isa_circuit, isa_observable, np.concatenate([x, params]).tolist()) for x in X_train]
            if simulation:
                res_tr = estimator.run(pubs_tr).result()
                evs_tr = [res_tr[i].data.evs for i in range(len(X_train))]
            else:
                res_tr = estimator.run([squash_pubs(pubs_tr)]).result()
                evs_tr = res_tr[0].data.evs
            losses_tr, preds_tr = [], []
            for i in range(len(X_train)):
                p1 = (1.0 + float(evs_tr[i])) / 2.0
                preds_tr.append(1 if p1 >= 0.5 else 0)
                losses_tr.append((p1 - y_train[i]) ** 2)

            pubs_te = [(isa_circuit, isa_observable, np.concatenate([x, params]).tolist()) for x in X_test]
            if simulation:
                res_te = estimator.run(pubs_te).result()
                evs_te = [res_te[i].data.evs for i in range(len(X_test))]
            else:
                res_te = estimator.run([squash_pubs(pubs_te)]).result()
                evs_te = res_te[0].data.evs
            preds_te = [1 if (1.0 + float(evs_te[i])) / 2.0 >= 0.5 else 0
                        for i in range(len(X_test))]

            return (float(np.mean(losses_tr)),
                    float(np.mean(np.array(preds_tr) == y_train)),
                    float(np.mean(np.array(preds_te) == y_test)))

        def batch_cost_func(params_list):
            indices = np.random.choice(len(X_train), size=min(BATCH_SIZE, len(X_train)), replace=False)
            X_batch, y_batch = X_train[indices], y_train[indices]
            M = len(X_batch)
            pubs = [
                (isa_circuit, isa_observable, np.concatenate([x, p]).tolist())
                for p in params_list for x in X_batch
            ]
            try:
                if simulation:
                    res = estimator.run(pubs).result()
                    evs_flat = [res[i].data.evs for i in range(len(pubs))]
                else:
                    res = estimator.run([squash_pubs(pubs)]).result()
                    evs_flat = list(res[0].data.evs)
                return [
                    float(np.mean([((1 + evs_flat[pi*M + s]) / 2 - y_batch[s]) ** 2 for s in range(M)]))
                    for pi in range(len(params_list))
                ]
            except Exception as e:
                raise RuntimeError(f"Batch estimator job failed: {e}") from e

        def train_optimizer(name, optimizer, initial_params, evals_per_call=min(BATCH_SIZE, len(X_train))):
            print_optimizer_start(name)
            np.random.seed(SEED)
            random.seed(SEED)
            gradient_strategy = getattr(optimizer, "grad_strategy", None)
            if gradient_strategy is not None and hasattr(gradient_strategy, "batch_eval"):
                gradient_strategy.batch_eval = batch_cost_func

            result, tracker, convergence_iteration, convergence_evals, convergence_time = run_with_tracking(
                optimizer, cost_func, initial_params,
                evals_per_call=evals_per_call,
                conv_tolerance=CONV_TOLERANCE)

            print(f"    [{name}] Evaluating accuracy", flush=True)
            _, train_acc, test_acc = train_and_evaluate(result.x)
            print(f"    Final Loss: {result.fun:.6f}  |  "
                  f"Train Acc: {train_acc:.2%}  |  Test Acc: {test_acc:.2%}", flush=True)

            report_result = ClassificationResult(
                optimizer_name=name,
                problem_name="Iris",
                final_loss=result.fun,
                loss_history=optimizer.history,
                train_acc=train_acc,
                test_acc=test_acc,
                train_acc_history=[],
                test_acc_history=[],
                noise_level=backend_label,
                wall_clock_time=tracker.wall_clock_time,
                circuit_evaluations=tracker.circuit_evaluations,
                eval_history=tracker.eval_history.copy(),
                time_history=tracker.time_history.copy(),
                convergence_iteration=convergence_iteration,
                convergence_evals=convergence_evals,
                convergence_time=convergence_time,
            )

            return attach_checkpoints(report_result, optimizer, CONV_TOLERANCE, evaluate=train_and_evaluate)

        all_optimizers_experiment(
            train_optimizer, initial_point, grad_free_iters,
            estimator, isa_circuit, isa_observable, X_train, y_train,
            save_prefix, title_suffix)


def main():
    return run_iris_experiment()


if __name__ == "__main__":
    main()
