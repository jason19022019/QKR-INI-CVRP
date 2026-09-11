"""Phase III: TSP routing for each repaired cluster.

Clusters with <=9 customers are solved exactly by brute force
(<=8! permutations); larger clusters use OR-Tools Guided Local
Search ([GLS], per Comment R1.4/general revision).
"""
import numpy as np
from itertools import permutations
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_tsp_cluster(member_ids, vrp_nodes, time_limit_seconds=1):
    if len(member_ids) == 1:
        c = member_ids[0]
        dist = 2 * np.sqrt(np.sum((vrp_nodes[0, :2] - vrp_nodes[c - 1, :2]) ** 2))
        return dist, member_ids

    node_indices = [0] + [c - 1 for c in member_ids]
    coords = vrp_nodes[node_indices, :2]
    N = len(coords)

    if N - 1 <= 9:
        # Exact: <=8! = 40320 permutations, effectively free and strictly
        # optimal, so there's no reason to hand this size to a metaheuristic.
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
        best_dist, best_perm = float('inf'), None
        for perm in permutations(range(1, N)):
            d = D[0][perm[0]] + sum(D[perm[i]][perm[i + 1]] for i in range(len(perm) - 1)) + D[perm[-1]][0]
            if d < best_dist:
                best_dist, best_perm = d, perm
        tour = [member_ids[i - 1] for i in best_perm]
        return best_dist, tour

    # [GLS] N-1 > 9: OR-Tools Guided Local Search. Distance matrix must be
    # integer for the routing solver, so it's built on a scaled copy;
    # reported distances are recomputed from the original float coords.
    scale = 1000
    D_int = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            D_int[i, j] = int(scale * np.sqrt(np.sum((coords[i] - coords[j]) ** 2)))

    manager = pywrapcp.RoutingIndexManager(N, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return D_int[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = int(time_limit_seconds)
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        total_dist = sum(
            np.sqrt(np.sum((coords[route[i]] - coords[route[i + 1]]) ** 2))
            for i in range(len(route) - 1)
        )
        total_dist += np.sqrt(np.sum((coords[route[-1]] - coords[0]) ** 2))
        tour = [member_ids[i - 1] for i in route[1:]]
        return total_dist, tour

    # Fallback only if GLS returns no solution at all within the time
    # budget (shouldn't happen in practice): same NN heuristic as before.
    visited = [0]
    remaining = set(range(1, N))
    while remaining:
        last = visited[-1]
        nxt = min(remaining, key=lambda x: D_int[last, x])
        visited.append(nxt)
        remaining.remove(nxt)
    visited.append(0)
    d = sum(np.sqrt(np.sum((coords[visited[i]] - coords[visited[i + 1]]) ** 2)) for i in range(N))
    tour = [member_ids[i - 1] for i in visited[1:-1]]
    return d, tour


def total_route_distance(repaired, customer_ids, vrp_nodes, time_limit_seconds=1):
    total = 0.0
    for mem in repaired:
        if len(mem) == 0:
            continue
        original_ids = [customer_ids[i] for i in mem]
        d, _ = solve_tsp_cluster(original_ids, vrp_nodes, time_limit_seconds=time_limit_seconds)
        total += d
    return total
