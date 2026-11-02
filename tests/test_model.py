import numpy as np
from imopso_uav3d.model import Problem, non_dominated, hypervolume_mc
from imopso_uav3d.optimizers import grid_ids, truncate_archive, swarm, evolutionary


def test_scenario_table():
    assert len(Problem(1).buildings) == 8
    assert len(Problem(2).buildings) == 13
    assert np.all(Problem(1).buildings[:, 2:] > 0)


def test_straight_line_objectives_and_dimensions():
    p = Problem()
    frac = np.linspace(0, 1, p.waypoints+2)[1:-1]
    x = p.start[None, :] + frac[:, None] * (p.goal-p.start)
    f = p.evaluate(x.flatten())
    assert f.shape == (1, 4)
    assert np.isclose(f[0, 0], 0)
    assert np.isclose(f[0, 2], 0)
    assert np.isclose(f[0, 3], 0, atol=1e-8)
    assert f[0, 1] >= 1


def test_non_dominance():
    f = np.array([[0, 1], [1, 0], [2, 2], [0.5, 0.5]])
    np.testing.assert_array_equal(non_dominated(f), [0, 1, 3])


def test_archive_capacity_and_dominance():
    rng = np.random.default_rng(4)
    x = rng.random((100, 6))
    f = rng.random((100, 4))
    ax, af = truncate_archive(x, f, 8, rng, True)
    assert 0 < len(ax) <= 8
    assert len(non_dominated(af)) == len(af)


def test_grid_boundary_and_hv():
    f = np.array([[0.1, 0.2], [0.6, 0.8], [0.9, 1.]])
    assert np.all((grid_ids(f) >= 0) & (grid_ids(f) < 10))
    s = np.random.default_rng(0).random((500, 4))
    assert hypervolume_mc([[0,0,0,0]], s) == 1.
    assert hypervolume_mc([[1,1,1,1]], s) == 0.


def test_smoke_all_four_algorithms():
    for method in ('imopso', 'mopso', 'nsga2', 'pesa2'):
        p = Problem(1, waypoints=4)
        rng = np.random.default_rng(7)
        kwargs = dict(particles=12, iterations=3, archive_size=8, interval=1,
                      mc_samples=np.random.default_rng(4).random((200, 4)))
        r = swarm(p, rng, improved=method=='imopso', **kwargs) if method in ('imopso', 'mopso') else evolutionary(p, rng, method=method, **kwargs)
        assert r.objectives.shape[1] == 4
        assert len(r.history) == 4
        assert np.isfinite(r.objectives).all()


def test_paper_inf_and_finite_numeric_collision_semantics():
    from imopso_uav3d.model import COLLISION_PENALTY
    p = Problem(scenario=1)
    a = np.linspace(0, 1, p.waypoints + 2)[1:-1, None]
    path = (p.start[None] + a * (p.goal - p.start)).flatten()
    assert p.evaluate(path)[0, 1] >= COLLISION_PENALTY
    assert np.isinf(p.evaluate_paper(path)[0, 1])


def test_continuous_collision_detector_and_short_segment():
    p = Problem(scenario=1, samples_per_segment=2)
    x = np.linspace(0, 1, p.waypoints + 2)[1:-1, None]
    direct = (p.start[None] + x * (p.goal - p.start)).flatten()
    assert p.exact_collision_mask(direct).tolist() == [True]


def test_paper_eq10_fixed_minmax_and_calibration():
    from imopso_uav3d.model import fixed_minmax, calibration_bounds
    p = Problem(1)
    lo, hi = calibration_bounds(p, size=1000, seed=1009)
    assert np.all(lo == 0)
    assert np.all(hi > lo)
    np.testing.assert_allclose(fixed_minmax(lo, lo, hi), np.zeros(4))
    np.testing.assert_allclose(fixed_minmax(hi, lo, hi), np.ones(4))


def test_replication_protocol_determinism():
    from imopso_uav3d.experiment import run
    _, a = run(1, 'imopso', iterations=5, particles=12, seed=71,
               archive_size=8, hv_samples=200, interval=1)
    _, b = run(1, 'imopso', iterations=5, particles=12, seed=71,
               archive_size=8, hv_samples=200, interval=1)
    np.testing.assert_array_equal(a.objectives, b.objectives)
    np.testing.assert_allclose(a.history, b.history)
