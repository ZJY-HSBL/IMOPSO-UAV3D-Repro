# Reproduction evidence (seed 42)

Run command:

    python -m imopso_uav3d.experiment --case both --algorithms imopso mopso nsga2 pesa2 --iterations 500 --particles 100 --archive-size 50 --seed 42 --hv-samples 6000 --interval 25 --output results

Run environment: independent Python/NumPy implementation, not the authors' MATLAB R2024a code.

The CSV contains MC-HV and final Pareto-archive mean objectives for eight algorithm/scenario combinations.

- Scenario 1 final HV: IMOPSO 0.9943, MOPSO 0.8620, NSGA-II 0.9918, PESA-II 0.8552.
- Scenario 2 final HV: IMOPSO 0.9927, MOPSO 0.9932, NSGA-II 0.9922, PESA-II 0.9918.

These are **single-seed independent reproductions**, not paper-reported data. In scenario 2, MOPSO narrowly exceeds IMOPSO. Do not use these figures to assert that paper results have been exactly reproduced.

Experimental PNGs and NPZ archives can be generated with the command above; full generated artefacts are also available in the companion downloadable ZIP for this reproduction.
