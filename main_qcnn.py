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
    Adam, BayesianOptimizer, CobylaOptimizer, FiniteDiffOptimizer,
    GradientDescentOCPClassifier, GradientDescentPSRClassifier,
    NelderMeadOptimizer, PGPEOptimizer, RMSprop, SGD, SPSAOptimizer,
)
from problems import QCNNClassifierProblem
from reporter import (print_category, print_optimizer_start,
                      print_qcnn_setup, print_qcnn_results_summary,
                      plot_qcnn_results, print_convergence_summary,
                      plot_convergence_metrics, save_convergence_report,
                      save_checkpoint_table)

simulation = True
BACKEND = "ibm_aachen"

SEED = 1234
NUM_IMAGES_PER_CLASS = 100
TEST_SIZE = 0.2
INIT_TYPE = "random"
EPOCHS = 100
MAX_ITER = 1000
BATCH_SIZE = 16
SHOTS = 256
CONV_TOLERANCE = 1e-4
EARLY_STOPPING = 10


def _make_rules():
    return {
        'SGD': SGD(learning_rate=0.05, decay=0.01),
        'Adam': Adam(lr=0.05, decay=0.01),
        'RMSprop': RMSprop(lr=0.05, decay=0.01),
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
    for opt_name, optimizer_cls in (
        ('FiniteDiff', FiniteDiffOptimizer),
        ('SPSA', SPSAOptimizer),
        ('PGPE', PGPEOptimizer),
    ):
        for rule_name, rule in _make_rules().items():
            results.append(train_optimizer(
                f"{opt_name}-{rule_name}",
                optimizer_cls(max_iter=EPOCHS, update_rule=rule,
                              early_stopping_patience=EARLY_STOPPING,
                              conv_tolerance=CONV_TOLERANCE),
                initial_point.copy()))

    print_category(3, "PARAMETER-SHIFT AND CURVATURE DIRECTIONS")
    for opt_name, optimizer_cls in (
        ('PSR', GradientDescentPSRClassifier),
        ('OCP', GradientDescentOCPClassifier),
    ):
        for rule_name, rule in _make_rules().items():
            kwargs = {'regularization_epsilon': 0.1} if opt_name == 'OCP' else {}
            results.append(train_optimizer(
                f"{opt_name}-{rule_name}",
                optimizer_cls(estimator, isa_circuit, isa_observable,
                              X_train, y_train, batch_size=BATCH_SIZE,
                              max_iter=EPOCHS, update_rule=rule,
                              early_stopping_patience=EARLY_STOPPING,
                              conv_tolerance=CONV_TOLERANCE, **kwargs),
                initial_point.copy()))

    for steps, checkpoint_results in checkpoint_groups(results):
        checkpoint_prefix = f"{save_prefix}_{steps}epochs_{SHOTS}shots"
        print(f"\nReporting checkpoint: {steps} recorded steps")
        best = print_qcnn_results_summary(checkpoint_results)
        plot_qcnn_results(checkpoint_results, best, n_epochs=steps, initialization=INIT_TYPE, shots=SHOTS, save_prefix=checkpoint_prefix)
        print_convergence_summary(checkpoint_results, tolerance_info=f"loss_diff < {CONV_TOLERANCE}")
        plot_convergence_metrics(checkpoint_results, save_prefix=checkpoint_prefix, title_suffix=title_suffix)
        save_convergence_report(checkpoint_results, save_prefix=checkpoint_prefix, tolerance_info=f"loss_diff < {CONV_TOLERANCE}")
        save_checkpoint_table(checkpoint_results, checkpoint_prefix, steps, shots=SHOTS)


def run_qcnn_experiment():
    np.random.seed(SEED)
    random.seed(SEED)

    problem = QCNNClassifierProblem(
        num_images_per_class=NUM_IMAGES_PER_CLASS,
        test_size=TEST_SIZE,
        seed=SEED,
    )

    if simulation:
        backend = AerSimulator(seed_simulator=SEED)
        session_ctx = contextlib.nullcontext()
        backend_label = "simulator"
        save_prefix, title_suffix = "qcnn", " — QCNN Classifier"
    else:
        service = QiskitRuntimeService(channel='ibm_quantum_platform')
        backend = service.backend(BACKEND)
        session_ctx = Session(backend=backend)
        backend_label = "hardware"
        save_prefix, title_suffix = "qcnn_hw", " — QCNN Classifier (Hardware)"

    print_qcnn_setup(backend.name, problem)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=SEED)
    isa_circuit = pm.run(problem.get_ansatz())
    isa_observable = problem.get_hamiltonian().apply_layout(isa_circuit.layout)

    print(f"  Transpiled depth        : {isa_circuit.depth()}")
    print(f"  Transpiled 2-qubit depth: "
          f"{isa_circuit.depth(lambda instr: len(instr.qubits) > 1)}")

    X_train, y_train = problem.X_train, problem.y_train
    X_test, y_test = problem.X_test, problem.y_test
    initial_point = problem.get_initial_point(initialization_type=INIT_TYPE)
    grad_free_iters = EPOCHS * max(1, len(X_train) // BATCH_SIZE)

    with session_ctx as active_session:
        estimator = Estimator(mode=backend) if simulation else Estimator(mode=active_session)
        estimator.options.default_shots = SHOTS
        if not simulation:
            estimator.options.resilience.measure_mitigation = True
            estimator.options.resilience.zne_mitigation = False
            estimator.options.resilience.pec_mitigation = False
            estimator.options.twirling.enable_gates = False
            estimator.options.twirling.enable_measure = True
            estimator.options.dynamical_decoupling.enable = False

        def training_loss(params, X, y):
            pubs = [(isa_circuit, isa_observable, np.concatenate([x, params]).tolist()) for x in X]
            if simulation:
                res = estimator.run(pubs).result()
                evs = [res[i].data.evs for i in range(len(X))]
            else:
                res = estimator.run([squash_pubs(pubs)]).result()
                evs = res[0].data.evs
            predictions = (1.0 + np.asarray(evs, dtype=float)) / 2.0
            return float(np.mean((predictions - y) ** 2))

        def cost_func(params):
            indices = np.random.choice(len(X_train), size=min(BATCH_SIZE, len(X_train)), replace=False)
            return training_loss(params, X_train[indices], y_train[indices])

        def batch_cost_func(params_list):
            indices = np.random.choice(len(X_train), size=min(BATCH_SIZE, len(X_train)), replace=False)
            X_batch, y_batch = X_train[indices], y_train[indices]
            pubs = [(isa_circuit, isa_observable, np.concatenate([x, p]).tolist())
                    for p in params_list for x in X_batch]
            if simulation:
                res = estimator.run(pubs).result()
                evs = [res[i].data.evs for i in range(len(pubs))]
            else:
                res = estimator.run([squash_pubs(pubs)]).result()
                evs = res[0].data.evs
            predictions = (1 + np.asarray(evs, dtype=float).reshape(len(params_list), len(X_batch))) / 2
            return np.mean((predictions - y_batch) ** 2, axis=1).tolist()

        def train_and_evaluate(params):
            pubs_tr = [(isa_circuit, isa_observable, np.concatenate([x, params]).tolist()) for x in X_train]
            pubs_te = [(isa_circuit, isa_observable, np.concatenate([x, params]).tolist()) for x in X_test]

            if simulation:
                res_tr = estimator.run(pubs_tr).result()
                evs_tr = [res_tr[i].data.evs for i in range(len(X_train))]
                res_te = estimator.run(pubs_te).result()
                evs_te = [res_te[i].data.evs for i in range(len(X_test))]
            else:
                res = estimator.run([squash_pubs(pubs_tr + pubs_te)]).result()
                evs_all = res[0].data.evs
                evs_tr = evs_all[:len(X_train)]
                evs_te = evs_all[len(X_train):]

            losses_tr, preds_tr = [], []
            for i in range(len(X_train)):
                p1 = (1.0 + float(evs_tr[i])) / 2.0
                preds_tr.append(1 if p1 >= 0.5 else 0)
                losses_tr.append((p1 - y_train[i]) ** 2)

            preds_te = [1 if (1.0 + float(evs_te[i])) / 2.0 >= 0.5 else 0
                        for i in range(len(X_test))]

            return (float(np.mean(losses_tr)),
                    float(np.mean(np.array(preds_tr) == y_train)),
                    float(np.mean(np.array(preds_te) == y_test)))

        def train_optimizer(name, optimizer, initial_params):
            print_optimizer_start(name)
            np.random.seed(SEED)
            random.seed(SEED)
            evals_per_call = min(BATCH_SIZE, len(X_train))
            gradient_strategy = getattr(optimizer, "grad_strategy", None)
            if gradient_strategy is not None and hasattr(gradient_strategy, "batch_eval"):
                gradient_strategy.batch_eval = batch_cost_func

            result, tracker, convergence_iteration, convergence_evals, convergence_time = run_with_tracking(
                optimizer, cost_func, initial_params,
                evals_per_call=evals_per_call,
                conv_tolerance=CONV_TOLERANCE)

            print(f"    [{name}] Evaluating accuracy", flush=True)
            best_index = int(np.argmin(optimizer.history))
            best_loss = float(optimizer.history[best_index])
            _, train_acc, test_acc = train_and_evaluate(optimizer.parameter_history[best_index])
            print(f"    Best Mini-batch Loss: {best_loss:.6f}  |  "
                  f"Train Acc: {train_acc:.2%}  |  Test Acc: {test_acc:.2%}", flush=True)

            report_result = ClassificationResult(
                optimizer_name=name,
                problem_name="QCNN-MNIST",
                final_loss=best_loss,
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

            return attach_checkpoints(
                report_result, optimizer, CONV_TOLERANCE,
                evaluate=train_and_evaluate)

        all_optimizers_experiment(
            train_optimizer, initial_point, grad_free_iters,
            estimator, isa_circuit, isa_observable, X_train, y_train,
            save_prefix, title_suffix)


def main():
    return run_qcnn_experiment()


if __name__ == "__main__":
    main()
