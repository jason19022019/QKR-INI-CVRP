# QKR-INI: Quantum Kernel Clustering + Quantum Knapsack Repair for CVRP

A hybrid quantum-classical pipeline for the Capacitated Vehicle Routing
Problem (CVRP):

- **Phase I** — quantum-kernel clustering (7 encodings: Amplitude, Angle,
  IQP, Z-Map, ZZ-Map, U3-Single, U3-TwoAngle), with a Nystrom
  low-rank approximation for instances beyond `--large-threshold` customers.
- **Phase II** — QKR-INI: capacity-feasibility repair via a QAOA-based
  0/1 knapsack subroutine (with a classical greedy fallback for
  sub-problems above `--max-qaoa-n`).
- **Phase III** — TSP routing per cluster (exact brute force for ≤9
  customers, OR-Tools Guided Local Search above that).
- **Refinement** — closed-loop Or-opt and/or ALNS destroy-and-repair
  inter-cluster search (`--refine-mode`).
- Classical baselines for comparison: Sweep, demand-aware K-Medoids,
  DBSCAN (all routed through the identical Phase II/III pipeline), and
  Clarke & Wright Savings (route-level).

## Installation

```bash
git clone https://github.com/<org>/qkr-ini.git
cd qkr-ini
pip install -e .
```

Requires Python ≥3.9. Dependencies (installed automatically): `numpy`,
`pandas`, `matplotlib`, `scikit-learn`, `pennylane`, `pennylane-lightning`,
`ortools`, `alns`, `jinja2`.

## Quick start

A tiny synthetic 8-customer instance is included for smoke-testing:

```bash
qkr-ini examples/tiny-example.vrp --n-seeds 3 --refine-mode both
```

or, without installing the console script:

```bash
python -m qkr_ini examples/tiny-example.vrp --n-seeds 3
```

Run `qkr-ini --help` for the full list of options (Nystrom landmark
count, penalty scaling, refinement mode, ALNS parameters, etc.).

## Benchmark instances used in the paper

This repository does not redistribute the full benchmark sets. Instances
are in standard CVRPLIB format (`.vrp`) and can be downloaded from:

- **Augerat et al. (1995), Sets A/B/E/P** — small-to-medium instances
  (16–100 customers): http://vrp.galgos.inf.puc-rio.br/index.php/en/
- **Christofides, Mingozzi & Toth (CMT) instances** — 50–199 customers:
  http://vrp.galgos.inf.puc-rio.br/index.php/en/
- **Uchoa et al. (2017) X-instances** — large-scale (100–1000 customers),
  used for the large-scale validation (up to X-n957-k87):
  http://vrp.galgos.inf.puc-rio.br/index.php/en/

All three families are hosted together on CVRPLIB
(http://vrp.galgos.inf.puc-rio.br/), which also lists best-known-solution
(BKS) values usable with `--bks`.

## Package layout

```
src/qkr_ini/
    cli.py                  CLI argument parsing
    data.py                 VRPLoader (CVRPLIB .vrp parser)
    feature_maps.py         quantum feature maps + kernel matrix (+ Nystrom)
    quantum_clustering.py   Phase I: quantum kernel k-means
    classical_clustering.py Sweep / K-Medoids / DBSCAN baselines
    repair.py               Phase II: QKR-INI (QAOA knapsack repair)
    routing.py              Phase III: TSP routing (brute force / OR-Tools GLS)
    refinement/
        or_opt.py            inter-cluster Or-opt local search
        alns_refine.py       ALNS destroy-and-repair refinement
    baselines/
        clarke_wright.py     Clarke & Wright Savings baseline
    pipeline.py              wires everything together (main())
```

## Reproducing paper results

```bash
qkr-ini path/to/CMT1.vrp --bks 524.61 --n-seeds 20 --refine-mode both
```

reports, per encoding/baseline: Silhouette/Davies-Bouldin/Calinski-Harabasz
clustering scores, QKR-INI feasibility and QAOA call count, pre- and
post-refinement route distance, gap to BKS, and a multi-seed
mean ± std reproducibility summary for the Angle encoding, plus a
LaTeX results table and two figures (`figure_phase1_convergence.pdf`,
`figure_quantum_vs_classical.pdf`).

## Citation

If you use this code, please cite the associated paper (citation to be
added on publication).
