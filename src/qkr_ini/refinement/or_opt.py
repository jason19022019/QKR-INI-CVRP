"""[R1-2] Closed-loop inter-cluster local search (Or-opt), run after
Phase III TSP costing."""
import numpy as np

from ..routing import solve_tsp_cluster

def inter_cluster_or_opt(clusters, capacity, customer_demands, customer_coords,
                          customer_ids, vrp_nodes, max_passes=8, neighbor_k=3,
                          tsp_time_limit=1):
    """
    Directly answers Reviewer #1's question: "What legitimate reason
    excludes closed-loop partition refinement after TSP cost evaluation?"
    None — this adds it. After Phase III produces routes, this performs
    an Or-opt style local search: relocate a single customer to a nearby
    cluster whenever doing so lowers the combined TSP cost of the two
    routes, subject to capacity. Restricted to each cluster's `neighbor_k`
    nearest centroids to keep this tractable on larger instances.

    [GLS] route_len calls solve_tsp_cluster, which now runs OR-Tools
    Guided Local Search for any trial cluster with >9 customers. Every
    candidate relocation evaluates TWO trial clusters (donor + receiver),
    so a single non-improving pass can issue on the order of
    max_passes * len(clusters) * cluster_size * neighbor_k * 2 GLS calls.
    tsp_time_limit is exposed here specifically so that cost can be
    tuned independently of the (likely larger) budget used for the
    one-off headline Phase III routing pass.
    """
    clusters = [list(c) for c in clusters]

    def route_len(mem):
        if not mem:
            return 0.0
        ids = [customer_ids[i] for i in mem]
        d, _ = solve_tsp_cluster(ids, vrp_nodes, time_limit_seconds=tsp_time_limit)
        return d

    costs = [route_len(c) for c in clusters]
    moves_made = 0
    for _ in range(max_passes):
        improved = False
        centroids = [np.mean(customer_coords[c], axis=0) if c else None for c in clusters]
        for a in range(len(clusters)):
            if centroids[a] is None:
                continue
            nbrs = sorted([b for b in range(len(clusters)) if b != a and centroids[b] is not None],
                          key=lambda b: np.linalg.norm(centroids[a] - centroids[b]))[:neighbor_k]
            for i, cust in enumerate(list(clusters[a])):
                for b in nbrs:
                    load_b = customer_demands[clusters[b]].sum() if clusters[b] else 0.0
                    if load_b + customer_demands[cust] > capacity:
                        continue
                    trial_a = clusters[a][:i] + clusters[a][i + 1:]
                    trial_b = clusters[b] + [cust]
                    nc_a, nc_b = route_len(trial_a), route_len(trial_b)
                    if nc_a + nc_b < costs[a] + costs[b] - 1e-9:
                        clusters[a], clusters[b] = trial_a, trial_b
                        costs[a], costs[b] = nc_a, nc_b
                        improved = True
                        moves_made += 1
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return clusters, sum(costs), moves_made
