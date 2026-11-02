"""Independent implementation of the 2026 Xu et al. four-objective UAV problem."""
from dataclasses import dataclass
import numpy as np

# Paper table 1: (cx, cy, width, depth); all buildings are 300 m high.
COLLISION_PENALTY = 1000.0  # Numeric surrogate of paper equation (4): +infinity.

BUILDINGS = np.array([
    [200, 300, 100, 100], [550, 250, 60, 60],
    [450, 550, 90, 90], [750, 600, 60, 60],
    [200, 500, 60, 60], [500, 800, 80, 80],
    [750, 400, 60, 60], [350, 120, 60, 60],
    [300, 800, 60, 60], [300, 650, 80, 80],
    [600, 120, 60, 60], [800, 250, 80, 80],
    [150, 850, 80, 80],
], dtype=float)


@dataclass(frozen=True)
class Problem:
    scenario: int = 1
    waypoints: int = 8
    z_min: float = 120.0
    z_max: float = 280.0
    drone_radius: float = 5.0
    safety_margin: float = 30.0
    samples_per_segment: int = 11

    def __post_init__(self):
        if self.scenario not in (1, 2):
            raise ValueError('scenario must be 1 or 2')
        if self.waypoints < 1 or self.samples_per_segment < 2:
            raise ValueError('invalid path discretization')
        if not (self.z_min < 200 < self.z_max):
            raise ValueError('this reference setup requires z_min < 200 < z_max')

    @property
    def buildings(self):
        return BUILDINGS[:8 if self.scenario == 1 else 13]

    @property
    def lower(self):
        return np.tile([0., 0., self.z_min], self.waypoints)

    @property
    def upper(self):
        return np.tile([1000., 1000., self.z_max], self.waypoints)

    @property
    def start(self):
        return np.array([0., 0., 200.])

    @property
    def goal(self):
        return np.array([1000., 1000., 200.])

    def paths(self, positions):
        x = np.asarray(positions, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.waypoints * 3:
            raise ValueError('expected (batch, waypoints*3) array')
        interior = x.reshape(-1, self.waypoints, 3)
        s = np.broadcast_to(self.start, (len(x), 1, 3))
        e = np.broadcast_to(self.goal, (len(x), 1, 3))
        return np.concatenate((s, interior, e), axis=1)

    def sample_positions(self, n, rng):
        # Smooth randomized lateral offsets are an *assumption*, not specified in the paper.
        frac = np.linspace(0., 1., self.waypoints + 2)[1:-1]
        base = self.start[None, :] + frac[:, None] * (self.goal - self.start)
        positions = np.repeat(base[None], n, axis=0)
        lateral = rng.uniform(-320., 320., (n, 1, 1)) * np.sin(np.pi * frac)[None, :, None]
        positions[:, :, :1] += lateral
        positions[:, :, 1:2] -= lateral
        positions[:, :, :2] += rng.normal(0., 55., (n, self.waypoints, 2))
        positions[:, :, 2] += rng.normal(0., 30., (n, self.waypoints))
        return np.clip(positions.reshape(n, -1), self.lower, self.upper)

    def exact_collision_mask(self, positions):
        """Continuous XY circle test, independent of the sampled paper objective.

        Returns a safety diagnostic, not a replacement for the paper's segment
        sampling equation. The paper's equivalent circle is used, not the drawn
        cuboid; its height does not affect the published obstacle objective.
        """
        p = self.paths(positions)
        a, b = p[:, :-1, :2], p[:, 1:, :2]
        d = b - a
        centers = self.buildings[:, :2]
        difference = centers[None, None, :, :] - a[:, :, None, :]
        t = np.clip(np.sum(difference*d[:, :, None, :], axis=-1) /
                    np.maximum(np.sum(d*d, axis=-1)[:, :, None], 1e-20), 0, 1)
        closest = a[:, :, None, :] + t[..., None]*d[:, :, None, :]
        dist = np.linalg.norm(closest-centers[None, None, :, :], axis=-1)
        radii = np.hypot(self.buildings[:, 2]/2, self.buildings[:, 3]/2)+self.drone_radius
        return np.any(dist < radii[None, None, :], axis=(1, 2))

    def evaluate_paper(self, positions):
        """Published objective convention: positive infinity at sampled collisions."""
        values = self.evaluate(positions)
        values[values[:, 1] >= COLLISION_PENALTY, 1] = np.inf
        return values

    def evaluate(self, positions):
        """f1-f4 (paper eqs. 2-8); invalid XY obstacles receive f2 >= 1.

        XY equivalent-radius collision matches eqs. (3)-(5); building height is not
        part of the published threat equation. Collision handling below is an
        explicit numerical convention, not a claim about authors' source code.
        """
        p = self.paths(positions)
        delta = np.diff(p, axis=1)
        distance = np.linalg.norm(delta, axis=2)
        total_length = distance.sum(axis=1)
        direct = np.linalg.norm(self.goal - self.start)
        f1 = 1. - direct / np.maximum(total_length, 1e-12)
        zideal = (self.z_min + self.z_max) / 2.
        f3 = np.mean(np.abs(p[:, 1:-1, 2] - zideal) / ((self.z_max - self.z_min) / 2.), axis=1)
        a = delta[:, :-1, :]
        b = delta[:, 1:, :]
        # arctan2 gives the correct angle in [0,pi], including obtuse turns.
        angles = np.arctan2(np.linalg.norm(np.cross(a, b), axis=2), np.sum(a*b, axis=2))
        f4 = np.mean(angles / np.pi, axis=1)

        # Uniform sampling along each segment, as stated below paper eq. (4).
        t = np.linspace(0., 1., self.samples_per_segment)
        points = p[:, :-1, None, :2] + t[None, None, :, None] * delta[:, :, None, :2]
        centers = self.buildings[:, :2]
        ds = np.linalg.norm(points[:, :, :, None, :] - centers[None, None, None, :, :], axis=-1)
        min_dist = ds.min(axis=2)
        radius = np.hypot(self.buildings[:, 2] / 2., self.buildings[:, 3] / 2.) + self.drone_radius
        clearance = min_dist - radius[None, None, :]
        risk = np.clip(1. - clearance / self.safety_margin, 0., 1.)
        f2 = risk.sum(axis=(1, 2)) / (self.waypoints + 1)
        collision = np.any(clearance < 0., axis=(1, 2))
        # Eq. (4) specifies +infinity at collisions. Store an explicit
        # finite infeasible penalty for stable Pareto, grid, and MC-HV handling.
        f2 = np.where(collision, COLLISION_PENALTY + np.maximum(0., -clearance).sum(axis=(1, 2)) / (self.waypoints + 1), f2)
        return np.column_stack((f1, f2, f3, f4))


def non_dominated(f):
    """Indices of Pareto non-dominated finite vectors (all goals minimized)."""
    f = np.asarray(f)
    if len(f) == 0:
        return np.empty(0, dtype=int)
    # is_dominated[i] iff a distinct row j dominates row i.
    dominates = np.all(f[:, None, :] <= f[None, :, :], axis=2) & np.any(
        f[:, None, :] < f[None, :, :], axis=2)
    return np.flatnonzero(~np.any(dominates, axis=0))


def hypervolume_mc(f, samples):
    """Fixed common-random-sample MC hypervolume, reference [1,1,1,1]."""
    f = np.asarray(f, dtype=float)
    f = f[np.all(np.isfinite(f) & (f < 1), axis=1)]
    if not len(f):
        return 0.
    return float(np.any(np.all(f[None, :, :] <= samples[:, None, :], axis=2), axis=1).mean())

def fixed_minmax(f, lo, hi):
    """Paper Eq.(10) with fixed shared objective bounds, frozen before algorithms.

    The authors do not disclose the bounds or when they are recomputed. A common
    frozen reference is needed for *comparable* between-algorithm HV curves.
    Values are deliberately not clipped: >1 solutions are dominated by the
    [1,1,1,1] HV reference and values <0 may indicate bad calibration bounds.
    """
    lo, hi = np.asarray(lo), np.asarray(hi)
    return (np.asarray(f) - lo) / np.maximum(hi - lo, 1e-12)


def calibration_bounds(problem, size=3000, seed=1009):
    """Common, algorithm-independent calibration population for Eq.(10).

    Infeasible samples are excluded from each objective's calibration range.
    The paper does not specify these calibration samples or normalization rules.
    """
    rng = np.random.default_rng(seed)
    samples = problem.evaluate(problem.sample_positions(size, rng))
    feasible = samples[samples[:, 1] < COLLISION_PENALTY]
    if len(feasible) < 20:
        raise ValueError('too few feasible calibration samples')
    lo = np.minimum(feasible.min(axis=0), 0)
    hi = np.maximum(feasible.max(axis=0), lo+1e-8)
    return lo, hi
