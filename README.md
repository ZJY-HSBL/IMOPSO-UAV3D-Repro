# IMOPSO-UAV3D-Repro

Independent executable reconstruction / 独立可运行复现.

Original paper: 徐航、廖运鹏、陈楚泽：《基于改进多目标粒子群优化算法的三维城区无人机路径规划》. 《莆田学院学报》, 2026, 33(5), 92–100.

## Implemented / 已实现

- Scenario 1: eight urban buildings; scenario 2: thirteen buildings (paper Table 1), 1 km × 1 km.
- Four objectives: excess path length, building threat, altitude offset, normalized turning angle (paper eqs. 2–8).
- Improved MOPSO: distance/iteration adaptive mutation, expanded hypergrid, adaptive leader choice, density- and singleton-aware archive management (eqs. 11–23).
- MOPSO, NSGA-II, PESA-II baselines, Monte Carlo hypervolume, Pareto archives, benchmark plots and CSV/JSON/NPZ export.
- Reproducible seeds and pytest smoke tests.

## Install & run / 安装与运行

    pip install -e ".[test]"
    pytest -q
    python -m imopso_uav3d.experiment --case both --iterations 500 --particles 100 --archive-size 50 --seed 42

Quick check:

    python -m imopso_uav3d.experiment --case both --iterations 35 --particles 45 --archive-size 30 --seed 2026 --hv-samples 2000 --interval 5

## Important reproduction limitations / 复现局限

This is an **independent Python reimplementation**, not the authors' original MATLAB source and **not yet a numerical replication of Table 2**. The article leaves multiple indispensable settings unspecified: path endpoints, number of waypoints, altitude limits, vehicle radius and safety distance, grid count, segment sampling count, mutation perturbation operator and random seeds. The defaults below are **explicit assumptions**:

- Start (0,0,200), goal (1000,1000,200) m; 8 intermediate points; altitude [120,280] m.
- Drone radius 5 m; safe margin 30 m; 11 threat samples per segment; grid divisions 10.
- The Gaussian mutation perturbation, ordinary MOPSO mutation probability 0.1, and genetic operators for NSGA-II/PESA-II are independently chosen implementation details.
- Paper section 4.1: PSO swarm 100, archive 50, omega initially 1 with factor 0.98, c1=c2=1.5; NSGA-II crossover/mutation 0.7/0.3; PESA-II 0.8/0.3.
- Paper eq. 4 uses two-dimensional equivalent-circle distance even though buildings are 3D; the implementation follows this formula. Infinite collision costs are represented by feasible-priority and finite penalty >=1 for computational stability.
- Angle uses atan2 for properly handling obtuse turns. Monte Carlo HV always uses the same sampling coordinates and [1,1,1,1] reference point for comparability.

The short smoke run is only a functionality test and **does not establish IMOPSO superiority**. Meaningful paper validation requires more seeds, confidence intervals, a harmonized evaluation budget and confirmation of unknown settings from the authors.

Third-party research reproduction; cite the original paper.