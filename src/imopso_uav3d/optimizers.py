"""MOPSO, IMOPSO, NSGA-II and region-based PESA-II baseline implementations."""
from dataclasses import dataclass
import numpy as np
from .model import non_dominated, hypervolume_mc, fixed_minmax, COLLISION_PENALTY


def grid_ids(f, bins=10, expand=0.1):
    f = np.asarray(f)
    low, high = f.min(axis=0), f.max(axis=0)
    span = np.maximum(high - low, 1e-10)
    lo = low - expand * span
    steps = (1 + 2 * expand) * span / bins
    return np.clip(np.floor((f - lo) / steps).astype(int), 0, bins - 1)


def group_cells(f, bins=10, expand=.1):
    ids = grid_ids(f, bins, expand)
    cells, inv, counts = np.unique(ids, axis=0, return_inverse=True, return_counts=True)
    return inv, counts


def choose_by_density(f, rng, pressure=1., archive_size=50, enhanced=False):
    inv, counts = group_cells(f)
    gamma = 1. + 0.1 * len(f) / archive_size if enhanced else 1.
    log_weights = -pressure * gamma * counts
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    cell = rng.choice(len(counts), p=weights)
    return int(rng.choice(np.flatnonzero(inv == cell)))


def truncate_archive(x, f, capacity, rng, enhanced):
    # Remove duplicate positions/objective vectors, then discard dominated rows.
    _, first = np.unique(np.round(f, 10), axis=0, return_index=True)
    x, f = x[np.sort(first)], f[np.sort(first)]
    # Prefer any collision-free paths over colliding paths.
    feasible = f[:, 1] < COLLISION_PENALTY
    if np.any(feasible):
        x, f = x[feasible], f[feasible]
    keep = non_dominated(f)
    x, f = x[keep], f[keep]
    while len(f) > capacity:
        inv, counts = group_cells(f)
        logw = 2. * counts + (np.log(.5) * (counts == 1) if enhanced else 0.)
        w = np.exp(logw - logw.max()); w /= w.sum()
        c = rng.choice(len(counts), p=w)
        k = rng.choice(np.flatnonzero(inv == c))
        x, f = np.delete(x, k, axis=0), np.delete(f, k, axis=0)
    return x, f


def preferred(old, new, rng):
    """Personal-best update with Pareto dominance and random ties."""
    new_wins = np.all(new <= old, axis=1) & np.any(new < old, axis=1)
    ties = ~(new_wins | (np.all(old <= new, axis=1) & np.any(old < new, axis=1)))
    return new_wins | (ties & (rng.random(len(old)) < 0.5))


@dataclass
class Result:
    decisions: np.ndarray
    objectives: np.ndarray
    history: list


def swarm(problem, rng, particles=100, iterations=500, archive_size=50,
          improved=True, interval=25, mc_samples=None, norm_bounds=None):
    x = problem.sample_positions(particles, rng)
    v = np.zeros_like(x)
    f = problem.evaluate(x)
    pbx, pbf = x.copy(), f.copy()
    ax, af = truncate_archive(x, f, archive_size, rng, improved)
    if mc_samples is None:
        mc_samples = np.random.default_rng(12345).random((6000, 4))
    hv = lambda arr: hypervolume_mc(fixed_minmax(arr, *norm_bounds) if norm_bounds is not None else arr, mc_samples)
    history = [(0, hv(af))]
    w = 1.
    for iteration in range(1, iterations + 1):
        leaders = np.stack([ax[choose_by_density(af, rng, archive_size=archive_size,
                          enhanced=improved)] for _ in range(particles)])
        v = w*v + 1.5*rng.random(x.shape)*(pbx-x) + 1.5*rng.random(x.shape)*(leaders-x)
        # Explicit unreported implementation choice: bound velocity to 20% of range.
        v = np.clip(v, -.2*(problem.upper-problem.lower), .2*(problem.upper-problem.lower))
        x = np.clip(x+v, problem.lower, problem.upper)
        if improved:
            # Paper eqs. (11)-(15), with p_mut interpreted as per-particle event.
            denom = np.maximum(np.abs(f), 1e-10)[:, None, :]
            d = np.sqrt(np.sum(((f[:, None, :] - af[None, :, :])/denom)**2, axis=2)).min(axis=1)
            dist_mod = np.minimum(1., d / .5)
            base = (1. - (iteration - 1) / max(iterations - 1, 1)) ** 2
            pmut = np.clip(base * (1. + dist_mod), .01, .8)
        else:
            pmut = np.full(particles, .1)  # Assumption: baseline mutation unspecified.
        mutated = rng.random(particles) < pmut
        if np.any(mutated):
            # Paper does not state actual perturbation distribution; Gaussian used.
            ix = np.flatnonzero(mutated)
            coordinate = rng.integers(0, x.shape[1], size=len(ix))
            x[ix, coordinate] += rng.normal(size=len(ix))*(problem.upper[coordinate]-problem.lower[coordinate])*.13
            x = np.clip(x, problem.lower, problem.upper)
        f = problem.evaluate(x)
        replace = preferred(pbf, f, rng)
        pbx[replace], pbf[replace] = x[replace], f[replace]
        ax, af = truncate_archive(np.vstack((ax, x)), np.vstack((af, f)), archive_size, rng, improved)
        w *= .98
        if iteration % interval == 0 or iteration == iterations:
            history.append((iteration, hv(af)))
    return Result(ax, af, history)


def fronts(f):
    remain = np.arange(len(f))
    layers = []
    while len(remain):
        feasible = f[remain, 1] < 1.
        subset = remain[feasible] if np.any(feasible) else remain
        selected = subset[non_dominated(f[subset])]
        layers.append(selected)
        remain = remain[~np.isin(remain, selected)]
    return layers


def crowding(f):
    n = len(f)
    if n < 3:
        return np.full(n, np.inf)
    crowd = np.zeros(n)
    for j in range(f.shape[1]):
        sort = np.argsort(f[:, j], kind='stable')
        crowd[sort[[0, -1]]] = np.inf
        span = max(f[sort[-1], j] - f[sort[0], j], 1e-12)
        crowd[sort[1:-1]] += (f[sort[2:], j] - f[sort[:-2], j]) / span
    return crowd


def nsga_elite(x, f, count):
    keep = []
    for layer in fronts(f):
        need = count - len(keep)
        if need <= 0:
            break
        if len(layer) > need:
            layer = layer[np.argsort(-crowding(f[layer]))[:need]]
        keep.extend(layer.tolist())
    return x[keep], f[keep]


def tournament_selection(f, rng, count):
    """NSGA-II binary tournament using nondominated rank and crowding distance."""
    ranks = np.empty(len(f), dtype=int)
    distances = np.empty(len(f), dtype=float)
    for rank, layer in enumerate(fronts(f)):
        ranks[layer] = rank
        distances[layer] = crowding(f[layer])
    pairs = rng.integers(0, len(f), size=(count, 2))
    a, b = pairs[:, 0], pairs[:, 1]
    choose_a = (ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) & (distances[a] > distances[b]))
    ties = (ranks[a] == ranks[b]) & (distances[a] == distances[b])
    choose_a = choose_a | (ties & (rng.random(count) < .5))
    return np.where(choose_a, a, b)


def genetic_step(x, rng, pc=.7, pm=.3, sigma=.08, lower=None, upper=None):
    """Uniform arithmetic crossover + single-coordinate Gaussian mutation.

    These genetic operators are stated as choices, not original authors' code.
    """
    n, d = x.shape
    order = rng.permutation(n)
    parents = x[order]
    children = parents.copy()
    for i in range(0, n-1, 2):
        if rng.random() < pc:
            blend = rng.random(d)
            children[i] = blend*parents[i] + (1-blend)*parents[i+1]
            children[i+1] = blend*parents[i+1] + (1-blend)*parents[i]
    mask = rng.random(n) < pm
    ids = np.flatnonzero(mask)
    if len(ids):
        col = rng.integers(0, d, len(ids))
        children[ids, col] += rng.normal(size=len(ids))*sigma*(upper[col]-lower[col])
    return np.clip(children, lower, upper)


def evolutionary(problem, rng, method='nsga2', particles=100, iterations=500,
                 archive_size=50, interval=25, mc_samples=None, norm_bounds=None):
    if method not in ('nsga2', 'pesa2'):
        raise ValueError('unknown evolutionary algorithm')
    if mc_samples is None:
        mc_samples = np.random.default_rng(12345).random((6000, 4))
    x = problem.sample_positions(particles, rng)
    f = problem.evaluate(x)
    ax, af = truncate_archive(x, f, archive_size, rng, True)
    hv = lambda arr: hypervolume_mc(fixed_minmax(arr, *norm_bounds) if norm_bounds is not None else arr, mc_samples)
    history = [(0, hv(af))]
    for iteration in range(1, iterations + 1):
        if method == 'nsga2':
            # Elitist selection from the previous iteration.
            parent_idx = tournament_selection(f, rng, particles)
            child = genetic_step(x[parent_idx], rng, .7, .3, lower=problem.lower, upper=problem.upper)
            cf = problem.evaluate(child)
            x, f = nsga_elite(np.vstack((x, child)), np.vstack((f, cf)), particles)
        else:
            # PESA-II: region-based parent selection from bounded elite archive.
            indices = [choose_by_density(af, rng, archive_size=archive_size)
                       for _ in range(particles)]
            child = genetic_step(ax[indices], rng, .8, .3,
                                 lower=problem.lower, upper=problem.upper)
            cf = problem.evaluate(child)
            x, f = child, cf
        ax, af = truncate_archive(np.vstack((ax, x)), np.vstack((af, f)), archive_size, rng, False)
        if iteration % interval == 0 or iteration == iterations:
            history.append((iteration, hv(af)))
    return Result(ax, af, history)
