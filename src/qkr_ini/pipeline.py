"""Main pipeline: wires together Phase I (quantum/classical clustering),
Phase II (QKR-INI repair), Phase III (TSP routing), closed-loop
refinement (Or-opt / ALNS), and the Clarke-Wright baseline, then
produces the paper's results table and figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .cli import parse_args
from .data import VRPLoader
from .feature_maps import compute_kernel_matrix, report_nystrom_quality
from .quantum_clustering import quantum_kernel_kmeans, plot_phase1_convergence
from .classical_clustering import sweep_clustering, demand_aware_kmedoids, dbscan_clustering
from .repair import qkr_ini_repair, greedy_repair
from .routing import total_route_distance
from .refinement import inter_cluster_or_opt, alns_refine_clusters
from .baselines import clarke_wright_savings

def main():
    args = parse_args()
    vrp = VRPLoader(args.vrp_file)

    depot_coord = vrp.nodes[0, :2]
    customer_coords = vrp.nodes[1:, :2]
    customer_demands = vrp.nodes[1:, 2]
    num_customers = len(customer_coords)
    customer_ids = list(range(2, 2 + num_customers))
    capacity = vrp.capacity
    n_clusters = int(np.ceil(np.sum(customer_demands) / capacity))

    print(f"Loaded {num_customers} customers. Total demand = {np.sum(customer_demands):.0f}, "
          f"K*Q = {n_clusters * capacity:.0f}, Clusters = {n_clusters}")
    if num_customers > args.large_threshold:
        print(f"[R1-1] Large instance detected (N={num_customers} > {args.large_threshold}); "
              f"Nystrom kernel approximation will be used for Phase I.")

    scaler = StandardScaler()
    norm_data = scaler.fit_transform(customer_coords)

    # ---------------- [R1-1] Standalone Nystrom validation (optional) -----
    if args.validate_nystrom:
        m_values = [int(x) for x in args.nystrom_m_values.split(",")]
        report_nystrom_quality(norm_data, args.nystrom_method, m_values,
                                seed=args.seed, large_threshold=args.large_threshold)
        print("\n[R1-1] --validate-nystrom requested: exiting after the "
              "convergence check (full pipeline was not run).")
        return None

    methods = ['Amplitude', 'Angle', 'IQP', 'Z-Map', 'ZZ-Map', 'U3-Single', 'U3-TwoAngle']

    # ---------------- Phase I: quantum kernel clustering ----------------
    cluster_labels, phase_one_stats, phase_one_history = {}, {}, {}
    print("\n=== Phase I: Quantum Kernel Clustering ===")
    for m in methods:
        print(f"Processing {m}...")
        K = compute_kernel_matrix(norm_data, m, args.large_threshold, args.n_landmarks, args.seed)
        labels, history = quantum_kernel_kmeans(K, n_clusters, max_iters=100, random_state=args.seed)
        cluster_labels[m] = labels
        phase_one_history[m] = history
        total_intra_dist, feasible_count = 0.0, 0
        for k in range(n_clusters):
            indices = np.where(labels == k)[0]
            centroid = np.mean(customer_coords[indices], axis=0)
            dist = np.sum(np.sqrt(np.sum((customer_coords[indices] - centroid) ** 2, axis=1)))
            total_intra_dist += dist
            if np.sum(customer_demands[indices]) <= capacity:
                feasible_count += 1
        phase_one_stats[m] = (total_intra_dist, feasible_count)
        status = "converged" if history["converged"] else "hit max_iters"
        print(f"  Intra-dist = {total_intra_dist:.1f}, Feasible = {feasible_count}/{n_clusters} "
              f"[{history['n_iters']} kernel k-means iters, {status}]")

    # [Convergence check requested] Phase I kernel k-means convergence plot
    plot_phase1_convergence(phase_one_history, methods, save_path="figure_phase1_convergence.pdf")

    # ---------------- [R1-3] Classical clustering baselines --------------
    print("\n=== [R1-3] Classical clustering baselines ===")
    classical_clusters = {
        'Sweep': sweep_clustering(customer_coords, customer_demands, capacity, depot_coord, n_clusters),
        'K-Medoids': demand_aware_kmedoids(customer_coords, customer_demands, capacity, n_clusters, seed=args.seed),
        'DBSCAN': dbscan_clustering(customer_coords, n_clusters, seed=args.seed),
    }
    for name, clu in classical_clusters.items():
        labels = np.zeros(num_customers, dtype=int)
        for k, mem in enumerate(clu):
            for i in mem:
                labels[i] = k
        cluster_labels[name] = labels
        total_intra_dist, feasible_count = 0.0, 0
        for k in range(n_clusters):
            indices = np.where(labels == k)[0]
            if len(indices) == 0:
                continue
            centroid = np.mean(customer_coords[indices], axis=0)
            dist = np.sum(np.sqrt(np.sum((customer_coords[indices] - centroid) ** 2, axis=1)))
            total_intra_dist += dist
            if np.sum(customer_demands[indices]) <= capacity:
                feasible_count += 1
        phase_one_stats[name] = (total_intra_dist, feasible_count)
        print(f"  {name}: Intra-dist = {total_intra_dist:.1f}, Feasible = {feasible_count}/{n_clusters}")

    all_methods = methods + list(classical_clusters.keys())

    # ---------------- Phase II: QKR-INI repair (all methods) -------------
    print("\n=== Phase II: QKR-INI Repair ===")
    phase_two_results, repaired_clusters_dict = {}, {}
    penalization_diagnostics = []
    for m in all_methods:
        labels = cluster_labels[m]
        clusters = [np.where(labels == k)[0].tolist() for k in range(n_clusters)]
        repaired, calls, _ = qkr_ini_repair(clusters, customer_coords, customer_demands, capacity,
                                             seed=args.seed, diagnostics=penalization_diagnostics,
                                             penalty_scale=args.penalty_scale, max_qaoa_n=args.max_qaoa_n)
        repaired_clusters_dict[m] = repaired
        total_dist, feasible_all = 0.0, True
        for mem in repaired:
            if len(mem) == 0:
                continue
            cent = np.mean(customer_coords[mem], axis=0)
            total_dist += np.sum(np.sqrt(np.sum((customer_coords[mem] - cent) ** 2, axis=1)))
            if np.sum(customer_demands[mem]) > capacity:
                feasible_all = False
        phase_two_results[m] = (total_dist, calls, feasible_all)
        print(f"{m}: intra-dist = {total_dist:.1f}, QAOA calls = {calls}, feasible = {feasible_all}")

    # [R2-4] Penalization diagnostic summary
    if penalization_diagnostics:
        n_misranked = sum(1 for d in penalization_diagnostics if d["infeasible_outranked_feasible"])
        pct = 100.0 * n_misranked / len(penalization_diagnostics)
        print(f"\n[R2-4] Unbalanced-penalization diagnostic (penalty_scale={args.penalty_scale}): "
              f"{n_misranked}/{len(penalization_diagnostics)} ({pct:.1f}%) QAOA sub-problems had an "
              f"infeasible bitstring score above the selected feasible one "
              f"(these calls fell back to the feasible-only selection rule, "
              f"so solution feasibility was never compromised).")

        # [Optimality diagnostic] How often does QAOA's returned bitstring
        # actually match the brute-force-exact optimum? (n <= max_n_exact
        # sub-problems only; larger ones print as skipped, not as failures.)
        checked = [d for d in penalization_diagnostics if d["exact_optimum"] is not None]
        skipped = len(penalization_diagnostics) - len(checked)
        if checked:
            gaps = [d["optimality_gap_pct"] for d in checked]
            n_exact_hit = sum(1 for g in gaps if g <= 1e-6)
            print(f"[Optimality] {n_exact_hit}/{len(checked)} sub-problems matched the exact "
                  f"brute-force optimum; mean gap = {np.mean(gaps):.2f}%, max gap = {np.max(gaps):.2f}% "
                  f"({skipped} sub-problem(s) skipped: n above the exact-check cutoff).")
        else:
            print(f"[Optimality] No sub-problems were small enough to brute-force check "
                  f"({skipped} skipped).")

    # ---------------- Classical greedy repair baseline (unchanged) -------
    greedy_dists = {}
    for m in methods:
        clusters = [np.where(cluster_labels[m] == k)[0].tolist() for k in range(n_clusters)]
        rep = greedy_repair(clusters, customer_coords, customer_demands, capacity)
        total_dist = 0.0
        for mem in rep:
            if len(mem) == 0:
                continue
            cent = np.mean(customer_coords[mem], axis=0)
            total_dist += np.sum(np.sqrt(np.sum((customer_coords[mem] - cent) ** 2, axis=1)))
        greedy_dists[m] = total_dist

    # ---------------- Phase III: TSP routing (all methods) ---------------
    print("\n=== Phase III: TSP Routing (all methods) ===")
    route_distances = {}
    for m in all_methods:
        route_distances[m] = total_route_distance(repaired_clusters_dict[m], customer_ids, vrp.nodes,
                                                    time_limit_seconds=args.tsp_time_limit)
        print(f"{m}: Total VRP distance = {route_distances[m]:.2f}")

    # ---------------- [R1-2] Inter-cluster Or-opt refinement -------------
    # [R1-2b support] The Or-opt-refined cluster partition (not just its
    # cost) is now captured in oropt_clusters_dict, so [R1-2b] ALNS below
    # can continue refining from it when --refine-mode=both.
    oropt_clusters_dict = {}
    refined_route_distances = {}
    if args.refine_mode in ("oropt", "both"):
        print("\n=== [R1-2] Closed-loop inter-cluster refinement (Or-opt) ===")
        for m in all_methods:
            refined_clusters, refined_total, moves = inter_cluster_or_opt(
                repaired_clusters_dict[m], capacity, customer_demands, customer_coords,
                customer_ids, vrp.nodes, tsp_time_limit=args.tsp_time_limit)
            oropt_clusters_dict[m] = refined_clusters
            refined_route_distances[m] = refined_total
            delta = route_distances[m] - refined_total
            print(f"{m}: {route_distances[m]:.2f} -> {refined_total:.2f} "
                  f"({moves} moves, saved {delta:.2f})")
    else:
        print("\n=== [R1-2] Or-opt refinement skipped (--refine-mode="
              f"{args.refine_mode}) ===")

    # ---------------- [R1-2b] ALNS destroy-and-repair refinement ---------
    alns_clusters_dict = {}
    alns_route_distances = {}
    if args.refine_mode in ("alns", "both"):
        print("\n=== [R1-2b] ALNS destroy-and-repair refinement ===")
        for m in all_methods:
            start_clusters = (oropt_clusters_dict[m] if args.refine_mode == "both"
                               else repaired_clusters_dict[m])
            baseline = (refined_route_distances[m] if args.refine_mode == "both"
                        else route_distances[m])
            refined_clusters, alns_total = alns_refine_clusters(
                start_clusters, capacity, customer_coords, customer_demands,
                depot_coord, customer_ids, vrp.nodes, seed=args.seed,
                num_iterations=args.alns_iterations,
                degree_of_destruction=args.alns_degree_of_destruction,
                max_string_removals=args.alns_max_string_removals,
                max_string_size=args.alns_max_string_size,
                tsp_time_limit=args.tsp_time_limit)
            alns_clusters_dict[m] = refined_clusters
            alns_route_distances[m] = alns_total
            delta = baseline - alns_total
            print(f"{m}: {baseline:.2f} -> {alns_total:.2f} (saved {delta:.2f})")

    def final_distance(m):
        """The route distance after whichever refinement stage(s) ran last."""
        if args.refine_mode == "none":
            return route_distances[m]
        if args.refine_mode == "oropt":
            return refined_route_distances[m]
        return alns_route_distances[m]  # 'alns' or 'both' -> ALNS ran last

    # ---------------- [R1-4] Clarke-Wright Savings baseline ---------------
    print("\n=== [R1-4] Clarke-Wright Savings (classical, route-level) ===")
    cw_routes, cw_total = clarke_wright_savings(depot_coord, customer_coords, customer_demands, capacity)
    print(f"Clarke-Wright: {len(cw_routes)} routes, total distance = {cw_total:.2f}")

    # ---------------- BKS gap reporting -----------------------------------
    BKS = args.bks
    if BKS:
        print(f"\nBest Known Solution (BKS) for this instance: {BKS}")
        for m in all_methods:
            gap = (final_distance(m) / BKS - 1) * 100
            print(f"{m}: Gap to BKS (post-refinement, mode={args.refine_mode}) = {gap:.1f}%")
        print(f"Clarke-Wright: Gap to BKS = {(cw_total / BKS - 1) * 100:.1f}%")

    # ---------------- [R2-9/10] Multi-seed reproducibility study ----------
    # [R2-10b] Now reports mean +/- std of route distance BOTH before and
    # after the selected closed-loop refinement stage(s) (--refine-mode),
    # so the reproducibility study also answers "does the refinement gain
    # hold up across seeds, or does it wash out under clustering/repair
    # noise?" -- not just "is the unrefined pipeline itself reproducible?".
    print(f"\n=== [R2-10] Reproducibility study: {args.n_seeds} seeds, "
          f"method='Angle', refine-mode={args.refine_mode} ===")

    def refine_pipeline(clusters, seed):
        """
        Applies whichever closed-loop refinement stage(s) --refine-mode
        selects to an arbitrary cluster partition, mirroring the main
        per-method [R1-2]/[R1-2b] pipeline above exactly, and returns the
        final exact route distance. Used so this study's "with
        refinement" numbers reflect the identical refinement pipeline,
        not a simplified stand-in.
        """
        current = [list(c) for c in clusters]
        final_dist = None
        if args.refine_mode in ("oropt", "both"):
            current, final_dist, _ = inter_cluster_or_opt(
                current, capacity, customer_demands, customer_coords,
                customer_ids, vrp.nodes, tsp_time_limit=args.tsp_time_limit)
        if args.refine_mode in ("alns", "both"):
            current, final_dist = alns_refine_clusters(
                current, capacity, customer_coords, customer_demands,
                depot_coord, customer_ids, vrp.nodes, seed=seed,
                num_iterations=args.alns_iterations,
                degree_of_destruction=args.alns_degree_of_destruction,
                max_string_removals=args.alns_max_string_removals,
                max_string_size=args.alns_max_string_size,
                tsp_time_limit=args.tsp_time_limit)
        if final_dist is None:  # refine_mode == "none": nothing ran above
            final_dist = total_route_distance(current, customer_ids, vrp.nodes,
                                               time_limit_seconds=args.tsp_time_limit)
        return final_dist

    seed_results_pre, seed_results_post = [], []
    for s in range(args.n_seeds):
        seed = args.seed + s
        K = compute_kernel_matrix(norm_data, 'Angle', args.large_threshold, args.n_landmarks, seed)
        labels = quantum_kernel_kmeans(K, n_clusters, max_iters=100, random_state=seed, track_history=False)
        clusters = [np.where(labels == k)[0].tolist() for k in range(n_clusters)]
        repaired, _, _ = qkr_ini_repair(clusters, customer_coords, customer_demands, capacity, seed=seed,
                                         penalty_scale=args.penalty_scale, max_qaoa_n=args.max_qaoa_n)
        dist_pre = total_route_distance(repaired, customer_ids, vrp.nodes,
                                         time_limit_seconds=args.tsp_time_limit)
        # No need to re-run an identical refinement pipeline when
        # --refine-mode=none -- "with" and "without" are the same number.
        dist_post = dist_pre if args.refine_mode == "none" else refine_pipeline(repaired, seed)

        seed_results_pre.append(dist_pre)
        seed_results_post.append(dist_post)
        print(f"  seed={seed}: pre-refine = {dist_pre:.2f}, "
              f"post-refine = {dist_post:.2f} (saved {dist_pre - dist_post:.2f})")

    seed_results_pre = np.array(seed_results_pre)
    seed_results_post = np.array(seed_results_post)
    print(f"\nAngle encoding over {args.n_seeds} seeds:")
    print(f"  Without closed-loop refinement: mean = {seed_results_pre.mean():.2f}, "
          f"std = {seed_results_pre.std():.2f}")
    print(f"  With closed-loop refinement ({args.refine_mode}): "
          f"mean = {seed_results_post.mean():.2f}, std = {seed_results_post.std():.2f}")
    if BKS:
        gap_pre = 100.0 * (seed_results_pre / BKS - 1)
        gap_post = 100.0 * (seed_results_post / BKS - 1)
        print(f"  Gap to BKS without refinement: mean = {gap_pre.mean():.1f}%, "
              f"std = {gap_pre.std():.1f}%")
        print(f"  Gap to BKS with refinement:    mean = {gap_post.mean():.1f}%, "
              f"std = {gap_post.std():.1f}%")

    # ---------------- Results table ---------------------------------------
    rows = []
    for m in all_methods:
        labels = cluster_labels[m]
        uniq = np.unique(labels)
        if len(uniq) > 1:
            sil = silhouette_score(norm_data, labels)
            db = davies_bouldin_score(norm_data, labels)
            ch = calinski_harabasz_score(norm_data, labels)
        else:
            sil = db = ch = np.nan
        d1 = phase_one_stats[m][0]
        d2, calls, feas = phase_two_results[m]
        vr = route_distances[m]
        vr_oropt = refined_route_distances.get(m)
        vr_alns = alns_route_distances.get(m)
        row = {
            'Method': m,
            'Type': 'Quantum' if m in methods else 'Classical',
            'Silhouette': f"{sil:.4f}" if not np.isnan(sil) else "N/A",
            'Davies-Bouldin': f"{db:.4f}" if not np.isnan(db) else "N/A",
            'Calinski-Harabasz': f"{ch:.2f}" if not np.isnan(ch) else "N/A",
            'Intra-dist (Stage 1)': f"{d1:.1f}",
            'Intra-dist (Repaired)': f"{d2:.1f}",
            'QAOA Calls': calls,
            'Feasible': 'Yes' if feas else 'No',
            'Route Dist (pre-refine)': f"{vr:.2f}",
            'Route Dist (post-Or-opt)': f"{vr_oropt:.2f}" if vr_oropt is not None else "N/A",
            'Route Dist (post-ALNS)': f"{vr_alns:.2f}" if vr_alns is not None else "N/A",
            'Route Dist (final)': f"{final_distance(m):.2f}",
        }
        if BKS:
            row['Gap to BKS (%)'] = f"{(final_distance(m) / BKS - 1) * 100:.1f}"
        rows.append(row)
    rows.append({
        'Method': 'Clarke-Wright', 'Type': 'Classical (route-level)',
        'Silhouette': 'N/A', 'Davies-Bouldin': 'N/A', 'Calinski-Harabasz': 'N/A',
        'Intra-dist (Stage 1)': 'N/A', 'Intra-dist (Repaired)': 'N/A', 'QAOA Calls': 0,
        'Feasible': 'Yes', 'Route Dist (pre-refine)': f"{cw_total:.2f}",
        'Route Dist (post-Or-opt)': 'N/A', 'Route Dist (post-ALNS)': 'N/A',
        'Route Dist (final)': f"{cw_total:.2f}",
        **({'Gap to BKS (%)': f"{(cw_total / BKS - 1) * 100:.1f}"} if BKS else {}),
    })

    df = pd.DataFrame(rows)
    latex_table = df.to_latex(
        index=False, escape=False,
        caption="Clustering quality, repair performance, and VRP routing distance across "
                "quantum feature maps, classical clustering baselines (Sweep, K-Medoids, DBSCAN), "
                "and the classical Clarke-Wright Savings route-level benchmark. "
                "'post-refine' applies the [R1-2] inter-cluster Or-opt local search.",
        label="tab:fullresults_revised",
        column_format="l" + "c" * (len(df.columns) - 1),
        float_format="%.2f"
    )
    print("\n=== LaTeX Table for Paper (revised) ===")
    print(latex_table)

    # ---------------- Figures ----------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
                          'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                          'lines.linewidth': 2, 'lines.markersize': 6})

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(all_methods) + 1)

    series = [('Phase III (pre-refine)', '#D55E00',
               [route_distances[m] for m in all_methods] + [cw_total])]
    if args.refine_mode in ("oropt", "both"):
        series.append(('Post Or-opt', '#0072B2',
                        [refined_route_distances[m] for m in all_methods] + [cw_total]))
    if args.refine_mode in ("alns", "both"):
        series.append(('Post ALNS', '#009E73',
                        [alns_route_distances[m] for m in all_methods] + [cw_total]))

    n_series = len(series)
    width = 0.8 / n_series
    offsets = [(-0.4 + width / 2 + i * width) for i in range(n_series)]
    for (label, color, values), offset in zip(series, offsets):
        ax.bar(x + offset, values, width, label=label, color=color,
               edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_methods + ['Clarke-Wright'], rotation=45, ha='right')
    ax.set_ylabel('Total VRP Route Distance')
    ax.set_title('Quantum vs. Classical Methods, Across Refinement Stages '
                  f'(--refine-mode={args.refine_mode})')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('figure_quantum_vs_classical.pdf', bbox_inches='tight')
    plt.close(fig)

    print("\nFigure saved: figure_quantum_vs_classical.pdf")
    print("Revised pipeline completed successfully.")
    return df
