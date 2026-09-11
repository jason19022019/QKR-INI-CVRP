"""[R1-4] Clarke & Wright (1964) Savings algorithm -- a route-level
(not cluster-first) classical CVRP metaheuristic baseline."""
import numpy as np

def clarke_wright_savings(depot, coords, demands, capacity):
    n = len(coords)
    d0 = np.linalg.norm(coords - depot, axis=1)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    routes = {i: [i] for i in range(n)}
    route_of = {i: i for i in range(n)}
    load = {i: demands[i] for i in range(n)}
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            savings.append((d0[i] + d0[j] - D[i, j], i, j))
    savings.sort(reverse=True, key=lambda x: x[0])
    for s, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        route_i, route_j = routes[ri], routes[rj]
        if route_i[0] != i and route_i[-1] != i:
            continue
        if route_j[0] != j and route_j[-1] != j:
            continue
        if load[ri] + load[rj] > capacity:
            continue
        if route_i[-1] != i:
            route_i = route_i[::-1]
        if route_j[0] != j:
            route_j = route_j[::-1]
        merged = route_i + route_j
        routes[ri] = merged
        load[ri] += load[rj]
        for node in merged:
            route_of[node] = ri
        del routes[rj]
        del load[rj]
    final_routes = list(routes.values())
    total = 0.0
    for r in final_routes:
        total += d0[r[0]] + sum(D[r[k], r[k + 1]] for k in range(len(r) - 1)) + d0[r[-1]]
    return final_routes, total
