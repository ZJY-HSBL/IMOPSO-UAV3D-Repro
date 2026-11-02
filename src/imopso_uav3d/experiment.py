"""Run both paper scenarios and export archives, paths, HV curves, and summaries."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .model import Problem, calibration_bounds, COLLISION_PENALTY
from .optimizers import swarm, evolutionary

METHODS = ('imopso', 'mopso', 'nsga2', 'pesa2')


def run(case=1, method='imopso', iterations=500, particles=100, seed=42,
        archive_size=50, hv_samples=6000, interval=25, waypoints=8):
    problem = Problem(scenario=case, waypoints=waypoints)
    rng = np.random.default_rng(seed)
    common_mc = np.random.default_rng(2026).random((hv_samples, 4))
    bounds = calibration_bounds(problem, size=3000, seed=1009)
    kwargs = dict(particles=particles, iterations=iterations, archive_size=archive_size,
                  interval=interval, mc_samples=common_mc, norm_bounds=bounds)
    if method == 'imopso':
        result = swarm(problem, rng, improved=True, **kwargs)
    elif method == 'mopso':
        result = swarm(problem, rng, improved=False, **kwargs)
    elif method in ('nsga2', 'pesa2'):
        result = evolutionary(problem, rng, method=method, **kwargs)
    else:
        raise ValueError('unknown method')
    return problem, result


def save(out, name, problem, result, seed):
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'{name}.npz', decisions=result.decisions,
                        objectives=result.objectives, history=np.array(result.history))
    summary = dict(name=name, seed=seed, scenario=problem.scenario, archive_size=len(result.objectives),
                   feasible_solutions=int(np.sum(result.objectives[:, 1] < COLLISION_PENALTY)),
                   continuously_safe_solutions=int(np.sum(~problem.exact_collision_mask(result.decisions))),
                   final_hv=result.history[-1][1], objective_means=result.objectives.mean(axis=0).tolist())
    (out / f'{name}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def plot_case(out, case, results, problem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for method, result in results.items():
        hist = np.array(result.history)
        ax.step(hist[:, 0], hist[:, 1], where='post', label=method.upper())
    ax.set(xlabel='Iteration', ylabel='Monte Carlo hypervolume',
           title=f'Scenario {case}: common reference (1,1,1,1)', ylim=(0, 1))
    ax.legend(); fig.tight_layout()
    fig.savefig(out / f'scenario_{case}_hv.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for building in problem.buildings:
        cx, cy, w, d = building
        ax.add_patch(Rectangle((cx-w/2, cy-d/2), w, d, facecolor='gray', alpha=.45))
    for method, result in results.items():
        feasible = result.objectives[:, 1] < COLLISION_PENALTY
        available = np.flatnonzero(feasible) if np.any(feasible) else np.arange(len(result.objectives))
        # Display feasible least-f1 archive member, not an arbitrary handpicked line.
        idx = available[np.argmin(result.objectives[available, 0])]
        points = problem.paths(result.decisions[idx])[0]
        ax.plot(points[:, 0], points[:, 1], marker='.', label=method.upper())
    ax.scatter([problem.start[0], problem.goal[0]], [problem.start[1], problem.goal[1]],
               marker='x', c='black', s=70)
    ax.set(xlim=(0, 1000), ylim=(0, 1000), xlabel='x (m)', ylabel='y (m)',
           title=f'Scenario {case}: shortest feasible Pareto paths')
    ax.set_aspect('equal'); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / f'scenario_{case}_paths.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['1','2','both'], default='both')
    parser.add_argument('--algorithms', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--particles', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--archive-size', type=int, default=50)
    parser.add_argument('--hv-samples', type=int, default=6000)
    parser.add_argument('--interval', type=int, default=25)
    parser.add_argument('--waypoints', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    cases = [1, 2] if args.case == 'both' else [int(args.case)]
    args.output.mkdir(exist_ok=True, parents=True)
    summaries = []
    for case in cases:
        results = {}
        for method in args.algorithms:
            problem, result = run(case, method, args.iterations, args.particles, args.seed,
                                  args.archive_size, args.hv_samples, args.interval, args.waypoints)
            name = f'scenario_{case}_{method}_seed_{args.seed}'
            summary = save(args.output, name, problem, result, args.seed)
            summaries.append(summary)
            results[method] = result
            print(f'{name}: feasible={summary["feasible_solutions"]}, MC-HV={summary["final_hv"]:.4f}', flush=True)
        plot_case(args.output, case, results, problem)
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'method', 'seed', 'archive_count', 'feasible', 'hv', 'f1', 'f2', 'f3', 'f4'])
        for v in summaries:
            method = v['name'].split('_')[2]
            writer.writerow([v['scenario'], method, v['seed'], v['archive_size'], v['feasible_solutions'],
                             v['final_hv'], *v['objective_means']])


if __name__ == '__main__':
    main()
