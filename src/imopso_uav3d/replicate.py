"""Auditable paired-seed experiments; independent of unpublished author settings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .model import Problem, calibration_bounds, COLLISION_PENALTY
from .experiment import run, save

METHODS = ('imopso', 'mopso', 'nsga2', 'pesa2')


def paired_intervals(paired, boot_seed=8842, reps=4000):
    """Exploratory paired bootstrap interval, not proof of significance for small n."""
    d = np.asarray(paired, float)
    rng = np.random.default_rng(boot_seed)
    if len(d) == 1:
        return [float(d[0]), float(d[0])]
    boot = d[rng.integers(len(d), size=(reps, len(d)))].mean(axis=1)
    return np.quantile(boot, [.025, .975]).tolist()


def experiment(seeds=(42, 43, 44), scenarios=(1, 2), iterations=500,
               particles=100, archive_size=50, hv_samples=12000,
               output=Path('rigorous_results')):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    comparisons = []
    config = dict(seeds=list(seeds), scenarios=list(scenarios), iterations=iterations,
                  particles=particles, archive_size=archive_size, hv_samples=hv_samples,
                  reference=[1, 1, 1, 1],
                  normalization='Paper Eq.10, shared fixed independent 3000-path calibration, seed 1009',
                  continuous_collision_check='exact XY segment/equivalent-circle intersection, reported separately',
                  provenance='independent reconstruction; authors did not publish full settings')
    for case in scenarios:
        p = Problem(case)
        lo, hi = calibration_bounds(p, 3000, 1009)
        config[f'case_{case}_objective_min'] = lo.tolist()
        config[f'case_{case}_objective_max'] = hi.tolist()
        for seed in seeds:
            for method in METHODS:
                problem, result = run(case, method, iterations, particles, seed,
                                      archive_size, hv_samples, max(1, iterations//20))
                label = f'scenario_{case}_{method}_seed_{seed}'
                save(output, label, problem, result, seed)
                raw = result.objectives
                sampled = raw[:, 1] < COLLISION_PENALTY
                continuous = ~problem.exact_collision_mask(result.decisions)
                stat = dict(scenario=case, algorithm=method, seed=seed, archive=len(raw),
                            sampled_feasible=int(sampled.sum()), continuously_safe=int(continuous.sum()),
                            mc_hv=float(result.history[-1][1]),
                            raw_f1_mean=float(raw[:, 0].mean()),
                            raw_f2_mean=float(raw[:, 1].mean()),
                            raw_f3_mean=float(raw[:, 2].mean()),
                            raw_f4_mean=float(raw[:, 3].mean()))
                records.append(stat)
                print(f'case={case} seed={seed} {method} HV_norm={stat["mc_hv"]:.5f} '
                      f'feasible={int(sampled.sum())}/{len(raw)} continuous={int(continuous.sum())}',
                      flush=True)
    with (output/'runs.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for case in scenarios:
        for method in METHODS[1:]:
            differences = []
            for seed in seeds:
                proposed = next(r for r in records if r['scenario'] == case
                                and r['algorithm'] == 'imopso' and r['seed'] == seed)
                baseline = next(r for r in records if r['scenario'] == case
                                and r['algorithm'] == method and r['seed'] == seed)
                differences.append(proposed['mc_hv'] - baseline['mc_hv'])
            ci = paired_intervals(differences)
            comparisons.append(dict(scenario=case, comparison=f'IMOPSO - {method.upper()}',
                                    n=len(differences),
                                    mean_hv_difference=float(np.mean(differences)),
                                    bootstrap_95ci_lower=ci[0],
                                    bootstrap_95ci_upper=ci[1],
                                    differences_by_seed=json.dumps(differences)))
    with (output/'paired_comparisons.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    (output/'protocol.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    return records, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    parser.add_argument('--scenarios', type=int, nargs='+', choices=[1,2], default=[1,2])
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--particles', type=int, default=100)
    parser.add_argument('--archive-size', type=int, default=50)
    parser.add_argument('--hv-samples', type=int, default=12000)
    parser.add_argument('--output', type=Path, default=Path('rigorous_results'))
    args = parser.parse_args()
    experiment(tuple(args.seeds), tuple(args.scenarios), args.iterations,
               args.particles, args.archive_size, args.hv_samples, args.output)

if __name__ == '__main__':
    main()
