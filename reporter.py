import csv
import os
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, MaxNLocator, PercentFormatter
from cycler import cycler


_METHOD_COLORS = {
    'COBYLA':      '#0072B2',
    'Nelder-Mead': '#D55E00',
    'Bayesian':    '#009E73',
    'FiniteDiff':  '#56B4E9',
    'SPSA':        '#CC79A7',
    'PGPE':        '#E69F00',
    'PSR':         '#333333',
    'OCP':         '#777744',
    'Adam':        '#0072B2',
    'SGD':         '#D55E00',
    'RMSprop':     '#009E73',
}

_METHOD_MARKERS = {
    'COBYLA':      'o',
    'Nelder-Mead': 's',
    'Bayesian':    '^',
    'FiniteDiff':  'D',
    'SPSA':        'v',
    'PGPE':        'P',
    'PSR':         '*',
    'OCP':         'X',
    'Adam':        'o',
    'SGD':         's',
    'RMSprop':     '^',
}

_RULE_STYLES = {
    'SGD':     '--',
    'Adam':    '-',
    'RMSprop': (0, (5, 1, 1, 1)),
    'Base':    '-',
}

_UPDATE_RULES = {'SGD', 'Adam', 'RMSprop'}

_CATEGORIES = {
    'Gradient-Free':          ['COBYLA', 'Nelder-Mead', 'Bayesian'],
    'Gradient Approximation': ['FiniteDiff', 'SPSA', 'PGPE'],
    'Analytical Gradient':    ['PSR', 'OCP'],
}


_SINGLE_COLUMN = (3.5, 2.6)
_DOUBLE_COLUMN = (7.2, 2.8)

CHEMICAL_ACCURACY_HA = 1.6e-3

_CONVERGENCE_CSV_FIELDS = [
    'optimizer', 'iterations', 'circuit_evaluations',
    'wall_clock_time_s', 'final_cost', 'convergence_iteration',
    'convergence_evals', 'convergence_time_s',
]


def _configure_plot_style():
    mpl.rcParams.update({
        'figure.dpi':           150,
        'figure.facecolor':     'white',
        'axes.facecolor':       'white',
        'font.family':          'sans-serif',
        'font.sans-serif':      ['DejaVu Sans'],
        'pdf.fonttype':         42,
        'ps.fonttype':          42,
        'svg.fonttype':         'none',
        'font.size':            9,
        'axes.titlesize':       9,
        'axes.labelsize':       9,
        'xtick.labelsize':      8,
        'ytick.labelsize':      8,
        'axes.linewidth':       0.8,
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.grid':            True,
        'axes.axisbelow':       True,
        'grid.color':           '#D8DDE3',
        'grid.linestyle':       '-',
        'grid.linewidth':       0.5,
        'grid.alpha':           0.55,
        'xtick.direction':      'out',
        'ytick.direction':      'out',
        'xtick.minor.visible':  False,
        'ytick.minor.visible':  False,
        'xtick.major.size':     4,
        'ytick.major.size':     4,
        'xtick.minor.size':     2,
        'ytick.minor.size':     2,
        'lines.linewidth':      1.5,
        'lines.markersize':     4,
        'legend.fontsize':      8,
        'legend.frameon':       False,
        'legend.framealpha':    0.9,
        'legend.fancybox':      False,
        'legend.edgecolor':     '#CCCCCC',
        'legend.borderpad':     0.4,
        'legend.labelspacing':  0.3,
        'axes.prop_cycle':      cycler('color', list(dict.fromkeys(_METHOD_COLORS.values()))),
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
        'savefig.pad_inches':   0.05,
    })


_configure_plot_style()


def _stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _split_optimizer_name(name: str):
    if '-' in name:
        method, rule = name.split('-', 1)
        if rule in _UPDATE_RULES:
            return method, rule
    return name, 'Base'


def _result_history(res):
    return getattr(res, 'history', getattr(res, 'loss_history', []))


def _final_value(res):
    hist = _result_history(res)
    return hist[-1] if hist else float('nan')


def _group_method_rules(results):
    methods = {}
    for res in results:
        method, rule = _split_optimizer_name(res.optimizer_name)
        if rule != 'Base':
            methods.setdefault(method, []).append((rule, res))
    return methods


def _build_params_str(n_epochs=None, initialization=None, shots=None):
    parts = []
    if n_epochs is not None:
        parts.append(f"Epochs: {n_epochs}")
    if initialization is not None:
        parts.append(f"Init: {initialization}")
    if shots is not None:
        parts.append(f"Shots: {shots}")
    return (" | " + " | ".join(parts)) if parts else ""


def _save_plot(fname):
    fig = plt.gcf()
    entries = {}
    for index, ax in enumerate(fig.axes):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            entries.setdefault(label, handle)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        title = ax.get_title()
        ax.set_title('')
        prefix = f'({chr(97 + index)}) ' if len(fig.axes) > 1 else ''
        width = 32 if len(fig.axes) > 1 else 48
        ax.set_title(textwrap.fill(prefix + title, width), loc='left',
                     fontsize=9, fontweight='medium', pad=9)
        ax.margins(x=0.03)
        if ax.get_yscale() == 'linear' and not isinstance(ax.yaxis.get_major_locator(), FixedLocator):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if ax.get_xlabel() in ('Recorded step', 'Objective evaluation', 'Circuit evaluations', 'Epoch'):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if entries:
        legend = fig.legend(
            entries.values(), entries.keys(), loc='lower center',
            bbox_to_anchor=(0.5, 0),
            ncol=min(len(entries), 3 if fig.get_figwidth() > 5 else 1),
            fontsize=8, frameon=False, handlelength=2.5,
            columnspacing=1.3, labelspacing=0.5)


        fig.canvas.draw()
        legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi + 0.12
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height + legend_height)
        bottom = legend_height / (height + legend_height)
        if fig.get_constrained_layout():
            fig.get_layout_engine().set(rect=(0, bottom, 1, 1 - bottom))
        else:
            fig.tight_layout(rect=(0, bottom, 1, 1))
    if not entries and not fig.get_constrained_layout():
        fig.tight_layout()
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf', '.svg'):
        output = path.with_suffix(suffix)
        fig.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.06)
    print(f"Saved plots: {path.with_suffix('')} (.png, .pdf, .svg)")
    plt.show()
    plt.close(fig)


def _method_color(res):
    method, _ = _split_optimizer_name(res.optimizer_name)
    return _METHOD_COLORS.get(method, '#333333')


def _method_marker(res):
    method, _ = _split_optimizer_name(res.optimizer_name)
    return _METHOD_MARKERS.get(method, 'o')


def _add_reference_lines(ax, ground_truth_cost, target_energy=None, optimal_cut=None):
    ax.axhline(y=ground_truth_cost, color='#888888', linestyle=':', linewidth=1.2,
               label='Initial energy', alpha=0.8)
    if target_energy is not None:
        ax.axhline(y=target_energy, color='#555555', linestyle=(0, (6, 2)), linewidth=1.5,
                   label='Target energy', alpha=0.9)


def _best_variants(results, target_energy=None):
    def sort_key(cost):
        return abs(cost - target_energy) if target_energy is not None else abs(cost)

    best = {}
    for result in results:
        method, rule = _split_optimizer_name(result.optimizer_name)
        final_value = _final_value(result)
        if method not in best or sort_key(final_value) < sort_key(best[method]['cost']):
            best[method] = {'result': result, 'cost': final_value, 'rule': rule}
    return best


def _plot_history(ax, result, label=None, style_name=None, x=None,
                  is_best=False, linewidth=1.5, alpha=0.8, markersize=4,
                  best_linewidth=2.0, best_alpha=0.9):
    method, rule = _split_optimizer_name(result.optimizer_name)
    style_name = style_name or method
    history = _result_history(result)
    x_values = range(1, len(history) + 1) if x is None else x
    ax.plot(
        x_values,
        history,
        label=label or result.optimizer_name,
        color=_METHOD_COLORS.get(style_name, '#333333'),
        linestyle=_RULE_STYLES.get(rule, '-'),
        linewidth=best_linewidth if is_best else linewidth,
        alpha=best_alpha if is_best else alpha,
        marker=_METHOD_MARKERS.get(style_name, 'o'),
        markevery=max(1, len(history) // 10),
        markersize=markersize,
    )


def _draw_loss_curves(ax, results, best_result, color_fn, title, ylabel="MSE Loss"):
    ax.set_title(title, fontweight='bold', pad=8)
    for res in results:
        _, rule = _split_optimizer_name(res.optimizer_name)
        is_best = res.optimizer_name == best_result.optimizer_name
        label = res.optimizer_name
        me = max(1, len(res.loss_history) // 10)
        ax.plot(res.loss_history, label=label,
                color=color_fn(res),
                linestyle=_RULE_STYLES.get(rule, '-'),
                linewidth=2.0 if is_best else 1.4,
                alpha=0.9 if is_best else 0.7,
                marker=_method_marker(res), markevery=me, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)


def _draw_accuracy_bars(ax, results, color_fn, title):
    names = [r.optimizer_name for r in results]
    accs  = [r.test_acc for r in results]
    y_pos = list(range(len(names)))
    bars  = ax.barh(y_pos, accs,
                    color=[color_fn(r) for r in results],
                    alpha=0.85, edgecolor='white', linewidth=0.5, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis='y', visible=False)
    ax.set_xlabel("Test Accuracy")
    ax.axvline(x=1.0, color='#555555', linestyle=':', linewidth=1.0, alpha=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1%}', va='center', fontsize=8)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)


def _plot_classifier_results(results, best_result, loss_title, acc_title,
                              fname, ylabel="MSE Loss", color_fn=None):
    if color_fn is None:
        color_fn = _method_color
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_DOUBLE_COLUMN, constrained_layout=True)
    loss_heading, _, metadata = loss_title.partition(' | ')
    acc_heading = acc_title.partition(' | ')[0]
    if metadata:
        fig.suptitle(metadata, fontsize=8, color='#555555')
    _draw_loss_curves(ax1, results, best_result, color_fn, loss_heading, ylabel)
    _draw_accuracy_bars(ax2, results, color_fn, acc_heading)
    _save_plot(fname)


def _print_classifier_summary(title, results):
    print(title)

    results_sorted = sorted(results, key=lambda r: r.test_acc, reverse=True)

    print(f"\n{'Rank':>4}  {'Optimizer':<22} {'Loss':>10} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 70)
    for i, res in enumerate(results_sorted, 1):
        print(f"{i:4d}. {res.optimizer_name:<22} {res.final_loss:10.6f} "
              f"{res.train_acc:9.2%} {res.test_acc:9.2%}")
    print("-" * 70)

    methods = _group_method_rules(results)
    if methods:
        print("\nBest update rule per method (test accuracy):")
        print("-" * 70)
        for method in sorted(methods.keys()):
            best_rule, best_res = max(methods[method], key=lambda x: x[1].test_acc)
            print(f"  {method:15s}: {best_rule:8s} | Test Acc: {best_res.test_acc:.2%}  "
                  f"Loss: {best_res.final_loss:.6f}")
        print("-" * 70)

    best = results_sorted[0]
    print(f"\n  Best overall: {best.optimizer_name}")
    print(f"  Test Accuracy : {best.test_acc:.2%}")
    print(f"  Train Accuracy: {best.train_acc:.2%}")
    print(f"  Final Loss    : {best.final_loss:.6f}")
    print(f"  Epochs        : {len(best.loss_history)}")
    return best


def print_setup(backend_name: str, reference_cost: float, optimal_cut: float = None):
    print(f"\nBackend: {backend_name}")
    print(f"Reference cost at initial point: {reference_cost:.6f}\n")
    if optimal_cut is not None:
        print(f"Brute Force Optimal Max-Cut Value: {optimal_cut}")
        print()


def print_category(category_num: int, category_name: str):
    print(f"Running Category {category_num}: {category_name}")


def print_optimizer_start(name: str):
    print(f"\nStarting optimizer: {name}")


def print_results_summary(results: list, ground_truth_cost: float, optimal_cut: float = None):
    print("Results closest to the target")

    target_energy = -optimal_cut if optimal_cut is not None else None

    def sort_key(r):
        return abs(r.final_cost - target_energy) if target_energy is not None else abs(r.final_cost)

    results_sorted = sorted(results, key=sort_key)

    print(f"\nGround truth reference cost: {ground_truth_cost:.6f}")
    if target_energy is not None:
        print(f"Brute Force Optimal Max-Cut:  {optimal_cut}")
        print(f"Target Energy (to minimize):  {target_energy:.6f}")
    print("\nRanking by distance to target:")
    print("-" * 70)

    for i, res in enumerate(results_sorted, 1):
        if target_energy is not None:
            dist = abs(res.final_cost - target_energy)
            pct = (dist / abs(target_energy) * 100) if target_energy != 0 else 0
            print(f"{i:2d}. {res.optimizer_name:20s} | Final: {res.final_cost:10.6f} | "
                  f"Distance to {target_energy:6.1f}: {dist:10.6f} ({pct:6.2f}%)")
        else:
            abs_cost = abs(res.final_cost)
            improvement = ((abs(ground_truth_cost) - abs_cost) / abs(ground_truth_cost) * 100
                           if ground_truth_cost != 0 else 0)
            print(f"{i:2d}. {res.optimizer_name:20s} | Final Cost: {res.final_cost:10.6f} | "
                  f"Abs: {abs_cost:10.6f} | Δ: {improvement:+7.2f}%")

    print("-" * 70)

    methods = _group_method_rules(results)
    if methods:
        print("\nBest update rule per gradient method:")
        print("-" * 70)
        for method in sorted(methods.keys()):
            rule, res = min(methods[method], key=lambda x: sort_key(x[1]))
            if target_energy is not None:
                print(f"  {method:15s}: {rule:8s} | Cost: {res.final_cost:10.6f} "
                      f"(Distance: {abs(res.final_cost - target_energy):.6f})")
            else:
                print(f"  {method:15s}: {rule:8s} | Cost: {res.final_cost:10.6f} "
                      f"(Abs: {abs(res.final_cost):.6f})")
        print("-" * 70)

    best = results_sorted[0]
    print(f"\n  Best overall: {best.optimizer_name}  {best.final_cost:.6f}")
    print(f"  Final Cost: {best.final_cost:.6f}")
    if target_energy is not None:
        dist = abs(best.final_cost - target_energy)
        pct = (dist / abs(target_energy) * 100) if target_energy != 0 else 0
        print(f"  Distance to Target {target_energy:.1f}: {dist:.6f} ({pct:.2f}%)")
    print(f"  Iterations: {len(best.history)}")
    return best


def plot_results_by_category(results, best_optimizer, ground_truth_cost,
                             optimal_cut=None, save_prefix="optimization"):
    target_energy = -optimal_cut if optimal_cut is not None else None
    best_per_method = _best_variants(results, target_energy)

    categories = [(name, methods) for name, methods in _CATEGORIES.items()
                  if any(method in best_per_method for method in methods)]
    if not categories:
        return
    size = _SINGLE_COLUMN if len(categories) == 1 else _DOUBLE_COLUMN
    fig, axes = plt.subplots(1, len(categories), figsize=size, sharey=True,
                             squeeze=False, constrained_layout=True)
    axes = axes[0]
    fig.suptitle('Best variant per method', fontsize=9)

    for ax, (cat_name, methods) in zip(axes, categories):
        ax.set_title(cat_name, fontweight='bold', pad=8)
        for method in methods:
            if method not in best_per_method:
                continue
            info = best_per_method[method]
            res  = info['result']
            rule = info['rule']
            label = method if rule == 'Base' else f"{method} ({rule})"
            _plot_history(ax, res, label=label,
                          is_best=res.optimizer_name == best_optimizer.optimizer_name,
                          markersize=4)
        _add_reference_lines(ax, ground_truth_cost, target_energy, optimal_cut)
        ax.set_xlabel("Recorded step")

    axes[0].set_ylabel("Cost (Energy)")
    _save_plot(f'images/{save_prefix}_by_category_{_stamp()}.png')


def plot_top3(results, ground_truth_cost, optimal_cut=None, save_prefix="optimization"):
    target_energy = -optimal_cut if optimal_cut is not None else None

    def sort_key(res):
        fv = _final_value(res)
        return abs(fv - target_energy) if target_energy is not None else abs(fv)

    top3 = sorted(results, key=sort_key)[:3]

    fig, ax = plt.subplots(figsize=_SINGLE_COLUMN, constrained_layout=True)
    ax.set_title(f"Top {len(top3)} optimizer" + ("s" if len(top3) != 1 else ""), fontweight='bold', pad=8)

    for rank, res in enumerate(top3, 1):
        _plot_history(ax, res, is_best=rank == 1,
                      linewidth=1.6, alpha=0.8, markersize=5,
                      best_linewidth=2.2, best_alpha=0.95)

    _add_reference_lines(ax, ground_truth_cost, target_energy, optimal_cut)
    ax.set_xlabel("Recorded step")
    ax.set_ylabel("Cost (Energy)")

    _save_plot(f'images/{save_prefix}_top3_{_stamp()}.png')


def plot_gradient_approx_detail(results, ground_truth_cost, optimal_cut=None, save_prefix="optimization"):
    target_energy = -optimal_cut if optimal_cut is not None else None
    grad_methods = _CATEGORIES['Gradient Approximation']

    method_results = {m: [] for m in grad_methods}
    for res in results:
        method, rule = _split_optimizer_name(res.optimizer_name)
        if method in method_results:
            method_results[method].append((rule, res))

    active_methods = [method for method in grad_methods if method_results[method]]
    if not active_methods:
        return
    size = _SINGLE_COLUMN if len(active_methods) == 1 else _DOUBLE_COLUMN
    fig, axes = plt.subplots(1, len(active_methods), figsize=size, sharey=True,
                             squeeze=False, constrained_layout=True)
    axes = axes[0]

    for ax, method in zip(axes, active_methods):
        ax.set_title(method, fontweight='bold', pad=8)
        for rule, res in sorted(method_results[method], key=lambda x: x[0]):
            _plot_history(ax, res, style_name=rule, markersize=4,
                          linewidth=1.5, alpha=0.85)
        _add_reference_lines(ax, ground_truth_cost, target_energy, optimal_cut)
        ax.set_xlabel("Recorded step")

    axes[0].set_ylabel("Cost (Energy)")
    _save_plot(f'images/{save_prefix}_gradient_approx_detail_{_stamp()}.png')


def print_iris_setup(backend_name: str, problem):
    print("Iris classifier")
    print(f"\nBackend: {backend_name}")
    problem.dataset_summary()
    print()


def print_iris_results_summary(results: list):
    return _print_classifier_summary("IRIS CLASSIFIER — RESULTS SUMMARY", results)


def plot_iris_results(results: list, best_result, n_epochs=None, initialization=None, shots=None, save_prefix="iris"):
    params_str = _build_params_str(n_epochs=n_epochs, initialization=initialization, shots=shots)
    _plot_classifier_results(
        results, best_result,
        loss_title="Training Loss (All Optimizers)" + params_str,
        acc_title="Final Test Accuracy" + params_str,
        fname=f'images/{save_prefix}_classifier_results_{_stamp()}.png',
    )


def print_qcnn_setup(backend_name: str, problem):
    print("MNIST QCNN (0 vs 1)")
    print(f"\nBackend: {backend_name}")
    problem.dataset_summary()
    print()


def print_qcnn_results_summary(results: list):
    return _print_classifier_summary("QCNN MNIST CLASSIFIER — RESULTS SUMMARY", results)


def plot_qcnn_results(results: list, best_result, n_epochs=None, initialization=None, shots=None, save_prefix="qcnn"):
    params_str = _build_params_str(n_epochs=n_epochs, initialization=initialization, shots=shots)
    _plot_classifier_results(
        results, best_result,
        loss_title="Training Loss — QCNN" + params_str,
        acc_title="Final Test Accuracy — QCNN" + params_str,
        fname=f'images/{save_prefix}_classifier_results_{_stamp()}.png',
    )


def print_h2_results_summary(results: list, exact_energy: float):
    print("Hydrogen VQE results")
    print(f"  Exact energy (FCI) : {exact_energy:.10f} Ha")
    print(f"  Chemical accuracy  : ±{CHEMICAL_ACCURACY_HA * 1e3:.1f} mHa  "
          f"(|E₀ − E_FCI| < {CHEMICAL_ACCURACY_HA:.4f} Ha)")

    results_sorted = sorted(results, key=lambda r: abs(r.final_cost - exact_energy))

    header = (f"  {'Rank':>4}  {'Optimizer':<24} {'Energy (Ha)':>14}  "
              f"{'Error (mHa)':>12}  {'Chem.Acc.':>9}  "
              f"{'Iters':>6}  {'Circ. Evals':>12}  {'Time (s)':>9}")
    sep = "  " + "-" * 88
    print(f"\n{header}")
    print(sep)

    for i, res in enumerate(results_sorted, 1):
        err_ha  = res.final_cost - exact_energy
        err_mha = err_ha * 1e3
        ca      = "Yes" if abs(err_ha) < CHEMICAL_ACCURACY_HA else "No"
        iters   = len(res.history)
        evals   = getattr(res, 'circuit_evaluations', 0)
        wtime   = getattr(res, 'wall_clock_time', 0.0)
        print(f"  {i:>4d}. {res.optimizer_name:<24} {res.final_cost:>14.8f}  "
              f"{err_mha:>+12.4f}  {ca:>9}  "
              f"{iters:>6}  {evals:>12,}  {wtime:>8.2f}s")

    print(sep)

    n_ca = sum(1 for r in results if abs(r.final_cost - exact_energy) < CHEMICAL_ACCURACY_HA)
    print(f"\n  {n_ca}/{len(results)} sampled energies lie within the reference threshold "
          f"(|error| < {CHEMICAL_ACCURACY_HA * 1e3:.1f} mHa).")

    best = results_sorted[0]
    best_err_mha = (best.final_cost - exact_energy) * 1e3
    ca_label = "within threshold" if abs(best_err_mha) < 1.6 else "outside threshold"
    print(f"\n  Best optimizer  : {best.optimizer_name}")
    print(f"  Final energy    : {best.final_cost:.10f} Ha")
    print(f"  Error from FCI  : {best_err_mha:+.4f} mHa  [{ca_label}]")
    print(f"  Iterations      : {len(best.history)}")
    print(f"  Circuit evals   : {getattr(best, 'circuit_evaluations', 0):,}")
    print(f"  Wall-clock time : {getattr(best, 'wall_clock_time', 0.0):.2f} s")
    return best


def plot_h2_energy_convergence(results: list, best, exact_energy: float,
                                save_prefix: str = "h2_vqe"):

    best_per_method = {}
    for res in results:
        method, rule = _split_optimizer_name(res.optimizer_name)
        fv = _final_value(res)
        if method not in best_per_method or fv < best_per_method[method]['cost']:
            best_per_method[method] = {'result': res, 'cost': fv, 'rule': rule}

    fig, ax = plt.subplots(figsize=_DOUBLE_COLUMN, constrained_layout=True)

    use_evaluations = all(
        len(getattr(info['result'], 'eval_history', [])) == len(info['result'].history)
        and len(info['result'].history) > 0 for info in best_per_method.values())

    ca = CHEMICAL_ACCURACY_HA
    ax.axhspan(exact_energy - ca, exact_energy + ca,
               alpha=0.13, color='#555555', zorder=1,
               label=f'Chemical accuracy (±{ca * 1e3:.1f} mHa)')
    ax.axhline(y=exact_energy, color='#555555', linestyle='--', linewidth=1.3,
               label='Exact energy (FCI)', alpha=0.9, zorder=2)

    for method, info in sorted(best_per_method.items()):
        res  = info['result']
        rule = info['rule']
        is_best = res.optimizer_name == best.optimizer_name
        label = method if rule == 'Base' else f"{method} ({rule})"
        me = max(1, len(res.history) // 12)
        x = res.eval_history if use_evaluations else range(1, len(res.history) + 1)
        ax.plot(x, res.history,
                label=label,
                color=_METHOD_COLORS.get(method, '#333333'),
                linestyle=_RULE_STYLES.get(rule, '-'),
                linewidth=2.0 if is_best else 1.4,
                alpha=0.95 if is_best else 0.75,
                marker=_METHOD_MARKERS.get(method, 'o'),
                markevery=me,
                markersize=4 if is_best else 3,
                zorder=3 if is_best else 2)

    ax.set_xlabel("Circuit evaluations" if use_evaluations else "Recorded step")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(r"H$_2$ ground-state energy (STO-3G, UCCSD)",
                 fontweight='bold', pad=10)

    _save_plot(f'images/{save_prefix}_energy_convergence_{_stamp()}.png')


def print_convergence_summary(results: list, tolerance_info: str = ""):
    print("Stability summary")
    if tolerance_info:
        print(f"  ({tolerance_info})")

    header = (f"{'Optimizer':<24} {'Final Cost':>12} {'Iters':>7} {'Circ Evals':>12} "
              f"{'Wall-Time(s)':>13} {'Stable Idx':>10} "
              f"{'Stable Evals':>11} {'Stable Time(s)':>13}")
    print(f"\n{header}")
    print("-" * 113)

    for res in results:
        ci = getattr(res, 'convergence_iteration', None)
        ce = getattr(res, 'convergence_evals', None)
        ct = getattr(res, 'convergence_time', None)
        fv = _final_value(res)
        print(f"{res.optimizer_name:<24} {fv:>12.6f} {len(_result_history(res)):>7} "
              f"{getattr(res, 'circuit_evaluations', 0):>12,} "
              f"{getattr(res, 'wall_clock_time', 0.0):>12.2f}s "
              f"{'N/A' if ci is None else ci:>10} "
              f"{'N/A' if ce is None else ce:>11} "
              f"{'N/A' if ct is None else f'{ct:.2f}':>13}")

    print("-" * 113)

    converged = [r for r in results if getattr(r, 'convergence_iteration', None) is not None]
    if converged:
        by_iter = min(converged, key=lambda r: r.convergence_iteration)
        by_eval = min(converged, key=lambda r: r.convergence_evals)
        by_time = min(converged, key=lambda r: r.convergence_time)
        print(f"\n  Earliest stability index : {by_iter.optimizer_name} ({_final_value(by_iter):.6f}) "
              f"(iter {by_iter.convergence_iteration})")
        print(f"  Fewest evaluations to stability: {by_eval.optimizer_name} ({_final_value(by_eval):.6f}) "
              f"({by_eval.convergence_evals:,} evals)")
        print(f"  Shortest time to stability  : {by_time.optimizer_name} ({_final_value(by_time):.6f}) "
              f"({by_time.convergence_time:.2f}s)")
    else:
        print("\n  No stability streak detected.")


def plot_convergence_metrics(results: list, save_prefix: str = "convergence",
                             title_suffix: str = "", target_cost: float = None):
    fig, axes = plt.subplots(1, 2, figsize=_DOUBLE_COLUMN, constrained_layout=True)
    if title_suffix:
        fig.suptitle(title_suffix.strip(' —–-'), fontsize=9)

    panels = [
        (axes[0], 'eval_history', "Circuit evaluations", "Evaluation budget"),
        (axes[1], 'time_history', "Wall-clock time (s)", "Elapsed time"),
    ]

    for ax, attr, xlabel, title in panels:
        ax.set_title(title, fontweight='bold', pad=8)
        for res in results:
            xh   = getattr(res, attr, [])
            hist = _result_history(res)
            if len(xh) == 0 or len(hist) == 0:
                continue
            n = min(len(xh), len(hist))
            method, rule = _split_optimizer_name(res.optimizer_name)
            color = _METHOD_COLORS.get(method, '#333333')
            ax.plot(xh[:n], hist[:n], label=res.optimizer_name,
                    color=color, linestyle=_RULE_STYLES.get(rule, '-'),
                    linewidth=1.3, alpha=0.9, marker=_method_marker(res),
                    markevery=max(1, n // 8), markersize=3)
        if target_cost is not None:
            ax.axhline(y=target_cost, color='#555555', linestyle=(0, (6, 2)), linewidth=1.5,
                       label='Target', alpha=0.85)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cost / Loss")

    _save_plot(f'images/{save_prefix}_convergence_{_stamp()}.png')


def save_convergence_report(results: list, save_prefix: str = "convergence",
                            tolerance_info: str = ""):
    stamp = _stamp()
    os.makedirs('images', exist_ok=True)

    csv_path = f'images/{save_prefix}_convergence_{stamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CONVERGENCE_CSV_FIELDS + [
            'first_small_change_iteration', 'first_small_change_evals',
            'first_small_change_time_s', 'stability_iteration',
            'stability_evals', 'stability_time_s'])
        writer.writeheader()
        for res in results:
            hist = _result_history(res)
            ct = getattr(res, 'convergence_time', None)
            writer.writerow({
                'optimizer':             res.optimizer_name,
                'iterations':            len(hist),
                'circuit_evaluations':   getattr(res, 'circuit_evaluations', 0),
                'wall_clock_time_s':     f"{getattr(res, 'wall_clock_time', 0.0):.4f}",
                'final_cost':            f"{hist[-1]:.6f}" if hist else 'N/A',
                'first_small_change_iteration': getattr(res, 'first_small_change_iteration', None),
                'first_small_change_evals': getattr(res, 'first_small_change_evals', None),
                'first_small_change_time_s': getattr(res, 'first_small_change_time', None),
                'stability_iteration': getattr(res, 'stability_iteration', None),
                'stability_evals': getattr(res, 'stability_evals', None),
                'stability_time_s': getattr(res, 'stability_time', None),
                'convergence_iteration': getattr(res, 'convergence_iteration', None),
                'convergence_evals':     getattr(res, 'convergence_evals', None),
                'convergence_time_s':    f"{ct:.4f}" if ct is not None else None,
            })
    print(f"  Saved convergence CSV: {csv_path}")

    def _conv_block(header, r):
        return [
            "", header,
            f"  Optimizer         : {r.optimizer_name} {_final_value(r):.6f}",
            f"  Stability Index  : {r.convergence_iteration}",
            f"  Circuit Evals     : {r.convergence_evals:,}",
            f"  Wall-Clock Time   : {r.convergence_time:.4f}s",
        ]

    converged = [r for r in results if getattr(r, 'convergence_iteration', None) is not None]
    lines = [
        f"Stability Report — {stamp}",
        f"Tolerance: {tolerance_info}",
        "=" * 80,
    ]
    if converged:
        lines += _conv_block("Earliest stability index",
                             min(converged, key=lambda r: r.convergence_iteration))
        lines += _conv_block("Fewest evaluations to stability",
                             min(converged, key=lambda r: r.convergence_evals))
        lines += _conv_block("Shortest time to stability",
                             min(converged, key=lambda r: r.convergence_time))
    else:
        lines += ["", "No stability streak detected."]

    best = min(results, key=_final_value)
    best_hist = _result_history(best)
    lines += [
        "", "=" * 80,
        "Terminal objective summary",
        f"  Optimizer         : {best.optimizer_name} {best_hist[-1]:.6f}",
        f"  Final Cost/Loss   : {best_hist[-1]:.6f}",
        f"  Total Iterations  : {len(best_hist)}",
        f"  Circuit Evals     : {getattr(best, 'circuit_evaluations', 0):,}",
        f"  Wall-Clock Time   : {getattr(best, 'wall_clock_time', 0.0):.4f}s",
        "=" * 80,
    ]

    txt_path = f'images/{save_prefix}_best_summary_{stamp}.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved summary TXT: {txt_path}")

    return csv_path, txt_path


def save_checkpoint_table(results, save_prefix, steps, shots=None):
    rows = []
    for result in results:
        history = _result_history(result)
        rows.append({
            "optimizer": result.optimizer_name,
            "checkpoint": steps,
            "recorded_steps": len(history),
            "shots": shots if shots is not None else "not recorded",
            "last_cost_or_loss": float(history[-1]),
            "best_cost_or_loss": float(min(history)),
            "train_accuracy": getattr(result, "train_acc", ""),
            "test_accuracy": getattr(result, "test_acc", ""),
            "circuit_evaluations": result.circuit_evaluations,
            "training_time_s": result.wall_clock_time,
            "first_small_change_iteration": getattr(result, "first_small_change_iteration", None),
            "stability_iteration": getattr(result, "stability_iteration", None),
        })
    if not rows:
        return
    path = Path("images") / f"{save_prefix}_table_{_stamp()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\hline",
             r"Optimizer & Steps & Last loss/cost & Best loss/cost & Train acc. & Test acc. & Evals & First small change & Stability \\",
             r"\hline"]
    for row in rows:
        name = row["optimizer"].replace("_", r"\_")
        train = f"{row['train_accuracy']:.4f}" if row['train_accuracy'] != "" else "--"
        test = f"{row['test_accuracy']:.4f}" if row['test_accuracy'] != "" else "--"
        lines.append(f"{name} & {row['recorded_steps']} & {row['last_cost_or_loss']:.6f} & "
                     f"{row['best_cost_or_loss']:.6f} & {train} & {test} & "
                     f"{row['circuit_evaluations']} & "
                     f"{row['first_small_change_iteration'] if row['first_small_change_iteration'] is not None else '--'} & "
                     f"{row['stability_iteration'] if row['stability_iteration'] is not None else '--'} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    metadata = f"% Checkpoint: {steps} recorded steps; shots: {shots}; accuracy in [0, 1]; diagnostic indices are zero-based.\n"
    path.with_suffix(".tex").write_text(metadata + "\n".join(lines) + "\n")
    print(f"  Saved checkpoint tables: {path}.csv and {path}.tex")
