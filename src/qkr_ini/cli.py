"""Command-line argument parsing for the QKR-INI pipeline."""
import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="qkr-ini",
        description="Quantum Knapsack Iterative-Repair pipeline for the "
                     "Capacitated Vehicle Routing Problem (CVRP)."
    )
    p.add_argument(
        "vrp_file",
        help="Path to a CVRPLIB-format .vrp instance file. See the "
             "examples/ directory and README for sample instances and "
             "links to the full benchmark sets (Augerat, CMT, X-series) "
             "used in the paper."
    )
    p.add_argument("--seed", type=int, default=42,
                    help="Base random seed (used for the single-run pipeline).")
    p.add_argument("--n-seeds", type=int, default=20,
                    help="[R2-10] Number of random seeds averaged for the "
                         "reproducibility study (mean +/- std of route cost).")
    p.add_argument("--large-threshold", type=int, default=100,
                    help="[R1-1] Customer count above which the Nystrom "
                         "kernel approximation is used instead of the exact "
                         "O(N^2) quantum kernel matrix.")
    p.add_argument("--n-landmarks", type=int, default=60,
                    help="[R1-1] Number of Nystrom landmark points for "
                         "large instances.")
    p.add_argument("--bks", type=float, default=None,
                    help="Best-known-solution value for gap reporting. If "
                         "omitted, gap-to-BKS rows are left as N/A.")
    p.add_argument("--penalty-scale", type=float, default=10,
                    help="[R2-4 experiment] Multiplies the QAOA knapsack's "
                         "default alpha/beta penalty weights. Sweep this "
                         "(e.g. 1.0, 2.0, 3.0) to see whether stronger "
                         "penalization reduces the infeasible-outranks-"
                         "feasible rate reported by the R2-4 diagnostic.")
    p.add_argument("--max-qaoa-n", type=int, default=12,
                    help="Single source of truth for 'too big to run QAOA "
                         "locally': sub-problems with more than this many "
                         "items use the classical greedy ratio heuristic "
                         "instead (both in qkr_ini_repair's Phase A/B calls "
                         "and as qaoa_knapsack's own internal fallback). "
                         "Cost on lightning.qubit grows steeply with n -- "
                         "~2s at n=12, ~5s at n=15, ~50s at n=18 per call "
                         "in our testing -- so raise this only after "
                         "benchmarking on your actual target machine.")
    p.add_argument("--validate-nystrom", action="store_true",
                    help="[R1-1 diagnostic] Run a standalone Nystrom-"
                         "approximation quality check (error vs. landmark "
                         "count) on this instance, save a convergence plot, "
                         "then exit WITHOUT running the full clustering/"
                         "repair/routing pipeline.")
    p.add_argument("--nystrom-m-values", type=str, default="10,20,30,50,80,120",
                    help="Comma-separated landmark counts to sweep for "
                         "--validate-nystrom.")
    p.add_argument("--nystrom-method", type=str, default="Angle",
                    help="Quantum feature map to validate the Nystrom "
                         "approximation against (used only with "
                         "--validate-nystrom).")
    p.add_argument("--tsp-time-limit", type=float, default=1.0,
                    help="[GLS] Per-call time budget (seconds) for OR-Tools "
                         "Guided Local Search in solve_tsp_cluster, used for "
                         "clusters with >9 customers (<=9 stays exact via "
                         "brute force). Applied to every call made from the "
                         "headline Phase III pass, the [R1-2] Or-opt "
                         "refinement loop, and the multi-seed reproducibility "
                         "study -- Or-opt alone can issue hundreds of calls "
                         "per method, so raising this trades runtime for "
                         "route quality faster than it looks.")
    p.add_argument("--refine-mode", type=str, default="alns",
                    choices=["oropt", "alns", "both", "none"],
                    help="[R1-2 / R1-2b] Which closed-loop inter-cluster "
                         "refinement stage(s) to run after Phase III "
                         "routing. 'oropt' = the original single-customer "
                         "relocate Or-opt search only. 'alns' = the new "
                         "ALNS destroy-and-repair search only (random + "
                         "SISR string removal, greedy repair), applied "
                         "directly to the Phase II clusters. 'both' "
                         "(default) runs Or-opt first, then lets ALNS "
                         "continue refining from the Or-opt-improved "
                         "partition -- a strictly larger neighbourhood "
                         "search than either alone. 'none' skips both and "
                         "reports unrefined Phase III distances.")
    p.add_argument("--alns-iterations", type=int, default=1000,
                    help="[R1-2b] ALNS destroy/repair iterations per "
                         "method. The classical CVRP-ALNS reference this "
                         "is adapted from used 3000 for a single instance; "
                         "that is almost certainly too slow once "
                         "multiplied across every quantum + classical "
                         "method and every (possibly large) X-instance in "
                         "this pipeline, so the default here is much "
                         "lower -- raise it if you have the wall-clock "
                         "budget.")
    p.add_argument("--alns-degree-of-destruction", type=float, default=0.10,
                    help="[R1-2b] Fraction of currently-routed customers "
                         "removed per random-removal destroy call.")
    p.add_argument("--alns-max-string-removals", type=int, default=3,
                    help="[R1-2b] Max number of clusters/routes touched "
                         "per SISR string-removal destroy call.")
    p.add_argument("--alns-max-string-size", type=int, default=10,
                    help="[R1-2b] Max customers removed per cluster/route "
                         "per SISR string-removal destroy call.")
    return p.parse_args(argv)
