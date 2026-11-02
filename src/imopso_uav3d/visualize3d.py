"""Render 3-D building obstacles and archived shortest feasible UAV routes."""
import argparse
from pathlib import Path
import numpy as np
from .model import Problem, COLLISION_PENALTY


def render(results_dir, scenario, seed=42, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    problem = Problem(scenario=scenario)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cx, cy, width, depth in problem.buildings:
        ax.bar3d(cx-width/2, cy-depth/2, 0, width, depth, 300,
                 color='slategray', alpha=.22, shade=True)
    plotted = []
    for method in ('imopso', 'mopso', 'nsga2', 'pesa2'):
        filename = results_dir / f'scenario_{scenario}_{method}_seed_{seed}.npz'
        if not filename.exists():
            continue
        data = np.load(filename)
        feasible = np.flatnonzero(data['objectives'][:, 1] < COLLISION_PENALTY)
        if not len(feasible):
            continue
        index = feasible[np.argmin(data['objectives'][feasible, 0])]
        pts = problem.paths(data['decisions'][index])[0]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2, label=method.upper())
        plotted.append(method)
    ax.scatter([problem.start[0], problem.goal[0]],
               [problem.start[1], problem.goal[1]],
               [problem.start[2], problem.goal[2]], color='black', marker='o', s=25)
    ax.set(xlim=(0,1000), ylim=(0,1000), zlim=(0,350), xlabel='x (m)',
           ylabel='y (m)', zlabel='z (m)', title=f'Scenario {scenario} - archived 3-D paths')
    if plotted:
        ax.legend()
    fig.tight_layout()
    output = Path(output) if output else results_dir / f'scenario_{scenario}_3d.png'
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--scenario', type=int, choices=[1,2], default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(render(args.results, args.scenario, args.seed, args.output))


if __name__ == '__main__':
    main()
