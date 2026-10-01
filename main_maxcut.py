import contextlib
import random

import numpy as np
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, QiskitRuntimeService, Session

from base import (
    OptimizationResult,
    attach_checkpoints,
    checkpoint_groups,
    run_with_tracking,
)
from optimisers import (
    Adam,
    BayesianOptimizer,
    CobylaOptimizer,
    FiniteDiffOptimizer,
    GradientDescentPSR,
    GradientDescentOCP,
    NelderMeadOptimizer,
    PGPEOptimizer,
    RMSprop,
    SGD,
    SPSAOptimizer,
)
from problems import MaxCutProblem
from reporter import (
    plot_convergence_metrics,
    plot_gradient_approx_detail,
    plot_results_by_category,
    plot_top3,
    print_category,
    print_convergence_summary,
    print_optimizer_start,
    print_results_summary,
    print_setup,
    save_convergence_report,
    save_checkpoint_table,
)


simulation = True
BACKEND = "ibm_aachen"
SEED = 1234
GRAPH_SEED = 42
INIT_STRATEGY = "random"
CONV_TOLERANCE = 1e-4
EPOCHS = 100
MAX_ITER = EPOCHS
SHOTS = 256


def _make_rules():
    return {
        'SGD': SGD(learning_rate=0.05),
        'Adam': Adam(lr=0.05),
        'RMSprop': RMSprop(lr=0.05),
    }


def all_optimizers_experiment(
    _tracked, initial_point,
    estimator, isa_ansatz, hamiltonian,
    save_prefix, title_suffix,
    ground_truth_cost, optimal_cut_value
):
    results = []

    print_category(1, "GRADIENT-FREE OPTIMIZERS")
    results.append(_tracked("COBYLA", CobylaOptimizer(max_iter=MAX_ITER), initial_point))
    results.append(_tracked("Nelder-Mead", NelderMeadOptimizer(max_iter=MAX_ITER), initial_point))
    results.append(_tracked("Bayesian", BayesianOptimizer(max_iter=MAX_ITER, n_initial_points=5), initial_point))

    print_category(2, "GRADIENT APPROXIMATIONS")
    gradient_classes = {
        'FiniteDiff': FiniteDiffOptimizer,
        'SPSA': SPSAOptimizer,
        'PGPE': PGPEOptimizer,
    }
    for opt_name, optimizer_cls in gradient_classes.items():
        for rule_name, rule in _make_rules().items():
            np.random.seed(SEED)
            random.seed(SEED)
            results.append(_tracked(
                f"{opt_name}-{rule_name}",
                optimizer_cls(max_iter=MAX_ITER, update_rule=rule, conv_tolerance=CONV_TOLERANCE),
                initial_point))

    print_category(3, "EXACT / ANALYTIC GRADIENTS")
    exact_classes = {
        'PSR': GradientDescentPSR,
        'OCP': GradientDescentOCP,
    }
    for opt_name, optimizer_cls in exact_classes.items():
        for rule_name, rule in _make_rules().items():
            np.random.seed(SEED)
            random.seed(SEED)
            results.append(_tracked(
                f"{opt_name}-{rule_name}",
                optimizer_cls(estimator, isa_ansatz, hamiltonian,
                         max_iter=MAX_ITER, update_rule=rule,
                         conv_tolerance=CONV_TOLERANCE),
                initial_point))

    for steps, checkpoint_results in checkpoint_groups(results):
        checkpoint_prefix = f"{save_prefix}_{steps}epochs_{SHOTS}shots"
        print(f"\nReporting checkpoint: {steps} recorded steps")
        best = print_results_summary(checkpoint_results, ground_truth_cost, optimal_cut_value)
        plot_results_by_category(checkpoint_results, best, ground_truth_cost, optimal_cut_value,
                                 save_prefix=checkpoint_prefix)
        plot_top3(checkpoint_results, ground_truth_cost, optimal_cut_value, save_prefix=checkpoint_prefix)
        plot_gradient_approx_detail(checkpoint_results, ground_truth_cost, optimal_cut_value,
                                    save_prefix=checkpoint_prefix)
        print_convergence_summary(checkpoint_results, tolerance_info=f"cost_diff < {CONV_TOLERANCE}")
        plot_convergence_metrics(checkpoint_results, save_prefix=checkpoint_prefix,
                                 title_suffix=title_suffix,
                                 target_cost=-optimal_cut_value)
        save_convergence_report(checkpoint_results, save_prefix=checkpoint_prefix,
                                tolerance_info=f"cost_diff < {CONV_TOLERANCE}")
        save_checkpoint_table(checkpoint_results, checkpoint_prefix, steps, shots=SHOTS)


def run_experiment():
    np.random.seed(SEED)
    random.seed(SEED)

    problem = MaxCutProblem(num_nodes=5, seed=GRAPH_SEED)
    hamiltonian = problem.get_hamiltonian()
    optimal_cut_value, _ = problem.brute_force_max_cut()

    if simulation:
        active_backend = AerSimulator(seed_simulator=SEED)
        session_ctx = contextlib.nullcontext()
        backend_label = "noiseless_sim"
        save_prefix, title_suffix = "maxcut_noiseless", " — MaxCut (Noiseless Sim)"
    else:
        service = QiskitRuntimeService(channel='ibm_quantum_platform')
        active_backend = service.backend(BACKEND)
        session_ctx = Session(backend=active_backend)
        backend_label = "hardware"
        save_prefix, title_suffix = "maxcut_hw", " — MaxCut (Hardware)"
    transpile_backend = active_backend

    print(f"Backend: {active_backend.name}\n")

    pm = generate_preset_pass_manager(backend=transpile_backend, optimization_level=3, seed_transpiler=SEED)
    isa_ansatz = pm.run(problem.get_ansatz())
    isa_hamiltonian = hamiltonian.apply_layout(isa_ansatz.layout)
    initial_point = problem.get_initial_point(initialization_type=INIT_STRATEGY)

    with session_ctx as active_session:
        estimator = Estimator(mode=active_backend if simulation else active_session)
        estimator.options.default_shots = SHOTS
        if not simulation:
            estimator.options.resilience.measure_mitigation = True
            estimator.options.resilience.zne_mitigation = False
            estimator.options.resilience.pec_mitigation = False
            estimator.options.twirling.enable_gates = False
            estimator.options.twirling.enable_measure = True
            estimator.options.dynamical_decoupling.enable = False


        def cost_func(params):
            pub = (isa_ansatz, isa_hamiltonian, np.array(params).tolist())
            try:
                return estimator.run([pub]).result()[0].data.evs
            except Exception as e:
                raise RuntimeError(f"Estimator job failed: {e}") from e

        def batch_cost_func(params_list):
            pubs = [(isa_ansatz, isa_hamiltonian, np.array(p).tolist()) for p in params_list]
            try:
                result = estimator.run(pubs).result()
                return [result[i].data.evs for i in range(len(pubs))]
            except Exception as e:
                raise RuntimeError(f"Batch estimator job failed: {e}") from e

        ground_truth_cost = cost_func(initial_point)
        print_setup(active_backend.name, ground_truth_cost, optimal_cut_value)

        def _tracked(name, optimizer, x0, evals_per_call=1):
            print_optimizer_start(name)
            np.random.seed(SEED)
            random.seed(SEED)
            gradient_strategy = getattr(optimizer, "grad_strategy", None)
            if gradient_strategy is not None:
                if hasattr(gradient_strategy, "batch_eval"):
                    gradient_strategy.batch_eval = batch_cost_func
                if hasattr(gradient_strategy, "cost_transform"):
                    gradient_strategy.cost_transform = lambda evs: evs
            result, tracker, convergence_iteration, convergence_evals, convergence_time = run_with_tracking(
                optimizer, cost_func, x0,
                evals_per_call=evals_per_call,
                conv_tolerance=CONV_TOLERANCE)
            report_result = OptimizationResult(
                name, "MaxCut", result.fun, optimizer.history, backend_label,
                wall_clock_time=tracker.wall_clock_time,
                circuit_evaluations=tracker.circuit_evaluations,
                eval_history=tracker.eval_history.copy(),
                time_history=tracker.time_history.copy(),
                convergence_iteration=convergence_iteration,
                convergence_evals=convergence_evals,
                convergence_time=convergence_time)

            return attach_checkpoints(report_result, optimizer, CONV_TOLERANCE)

        all_optimizers_experiment(
            _tracked, initial_point,
            estimator, isa_ansatz, hamiltonian,
            save_prefix, title_suffix,
            ground_truth_cost, optimal_cut_value)


def main():
    return run_experiment()


if __name__ == "__main__":
    main()
