"""[R1-2b] ALNS-based closed-loop destroy-and-repair refinement.

Named `alns_refine` (not `alns`) to avoid shadowing the third-party
`alns` package (PyPI: alns) that this module imports from.
"""
import functools
import numpy as np
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations

from ..utils import eucl_dist
from ..routing import solve_tsp_cluster


def cluster_route_cost(route, customer_coords, depot_coord):
    """
    Euclidean tour length for an explicitly ORDERED route (list of
    0-indexed customer indices), depot -> route -> depot. This is the
    fast surrogate objective ALNS searches against; final reported costs
    still go through the exact solve_tsp_cluster (GLS/brute-force) used
    elsewhere in the pipeline.
    """
    if not route:
        return 0.0
    pts = [depot_coord] + [customer_coords[c] for c in route] + [depot_coord]
    return sum(eucl_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


class ClusterAlnsState:
    """
    ALNS solution state for the [R1-2b] refinement stage. Mirrors
    `CvrpState` from the reference CVRP-ALNS notebook: `routes` is a list
    of explicitly ORDERED customer-index lists (one per cluster/vehicle
    route), `unassigned` holds customers currently removed by a destroy
    operator and awaiting repair.
    """

    def __init__(self, routes, customer_coords, depot_coord, unassigned=None):
        self.routes = routes
        self.customer_coords = customer_coords
        self.depot_coord = depot_coord
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        return ClusterAlnsState(
            [list(r) for r in self.routes], self.customer_coords,
            self.depot_coord, list(self.unassigned),
        )

    def objective(self):
        return sum(cluster_route_cost(r, self.customer_coords, self.depot_coord)
                    for r in self.routes)

    @property
    def cost(self):
        return self.objective()

    def find_route(self, customer):
        for route in self.routes:
            if customer in route:
                return route
        raise ValueError(f"Customer {customer} not found in any route.")


# ---------------------- Destroy operators ----------------------

def alns_random_removal(state, rng, degree_of_destruction):
    """Removes a random subset of currently-routed customers."""
    destroyed = state.copy()
    all_customers = [c for r in destroyed.routes for c in r]
    if not all_customers:
        return destroyed

    n_remove = max(1, min(len(all_customers),
                           int(len(all_customers) * degree_of_destruction)))
    chosen = rng.choice(all_customers, size=n_remove, replace=False)
    for c in chosen:
        c = int(c)
        route = destroyed.find_route(c)
        route.remove(c)
        destroyed.unassigned.append(c)

    destroyed.routes = [r for r in destroyed.routes if r]
    return destroyed


def _nearest_customers(seed_idx, customer_coords):
    d = np.sqrt(np.sum((customer_coords - customer_coords[seed_idx]) ** 2, axis=1))
    order = np.argsort(d)
    return [int(i) for i in order if i != seed_idx]


def _remove_string(route, cust, max_string_size, rng):
    """Removes a contiguous (wrap-around) string of the route containing `cust`."""
    size = int(rng.integers(1, min(len(route), max_string_size) + 1))
    start = route.index(cust) - int(rng.integers(size))
    idcs = sorted({idx % len(route) for idx in range(start, start + size)}, reverse=True)
    return [route.pop(idx) for idx in idcs]


def alns_string_removal(state, rng, customer_coords, max_string_removals, max_string_size):
    """
    Simplified SISR (Christiaens & Vanden Berghe, 2020): removes
    contiguous strings of customers from routes near a randomly chosen
    seed customer, instead of isolated random customers -- this tends to
    open up more "reconnectable" gaps for the repair operator than pure
    random removal.
    """
    destroyed = state.copy()
    all_customers = [c for r in destroyed.routes for c in r]
    if not all_customers:
        return destroyed

    avg_route_size = max(1, int(np.mean([len(r) for r in destroyed.routes])))
    eff_max_string_size = max(max_string_size, avg_route_size)
    max_removals = min(len(destroyed.routes), max_string_removals)

    center = int(rng.choice(all_customers))
    touched_routes = []

    for cust in _nearest_customers(center, customer_coords):
        if len(touched_routes) >= max_removals:
            break
        try:
            route = destroyed.find_route(cust)
        except ValueError:
            continue  # already removed by an earlier string this call
        if route in touched_routes:
            continue
        removed = _remove_string(route, cust, eff_max_string_size, rng)
        destroyed.unassigned.extend(removed)
        touched_routes.append(route)

    destroyed.routes = [r for r in destroyed.routes if r]
    return destroyed


# ---------------------- Repair operator ----------------------

def alns_best_insert(cust, state, customer_demands, capacity):
    """Cheapest feasible (route, position) to insert `cust` at, by Euclidean insertion delta."""
    best_cost, best_route, best_idx = None, None, None
    for route in state.routes:
        load = customer_demands[route].sum() if route else 0.0
        if load + customer_demands[cust] > capacity:
            continue
        for idx in range(len(route) + 1):
            pred = state.depot_coord if idx == 0 else state.customer_coords[route[idx - 1]]
            succ = state.depot_coord if idx == len(route) else state.customer_coords[route[idx]]
            added = (eucl_dist(pred, state.customer_coords[cust])
                     + eucl_dist(state.customer_coords[cust], succ)
                     - eucl_dist(pred, succ))
            if best_cost is None or added < best_cost:
                best_cost, best_route, best_idx = added, route, idx
    return best_route, best_idx


def alns_greedy_repair(state, rng, customer_demands, capacity):
    order = list(state.unassigned)
    rng.shuffle(order)
    state.unassigned = []
    for cust in order:
        route, idx = alns_best_insert(cust, state, customer_demands, capacity)
        if route is not None:
            route.insert(idx, cust)
        else:
            # No feasible existing route (shouldn't normally happen, since
            # capacity is respected by construction) -- open a new route.
            state.routes.append([cust])
    return state


# ---------------------- Driver ----------------------

def alns_refine_clusters(clusters, capacity, customer_coords, customer_demands,
                          depot_coord, customer_ids, vrp_nodes, seed=0,
                          num_iterations=1000, degree_of_destruction=0.10,
                          max_string_removals=3, max_string_size=15,
                          tsp_time_limit=1):
    """
    [R1-2b] Runs an ALNS destroy-and-repair search over the given cluster
    partition, then re-costs the final partition exactly via
    solve_tsp_cluster (matching Phase III / [R1-2] reporting). Returns
    (refined_clusters, exact_total_distance).
    """
    clusters = [list(c) for c in clusters if len(c) > 0]
    if not clusters:
        return [], 0.0

    # Seed each route with an explicit visiting order via the exact TSP
    # solver, so the Euclidean surrogate objective used during search
    # starts from a real (near-)optimal tour rather than raw index order.
    ordered_routes = []
    for mem in clusters:
        ids = [customer_ids[i] for i in mem]
        _, tour = solve_tsp_cluster(ids, vrp_nodes, time_limit_seconds=tsp_time_limit)
        ordered_routes.append([t - 2 for t in tour])  # customer_ids[i] == i + 2

    state = ClusterAlnsState(ordered_routes, customer_coords, depot_coord)

    rng = np.random.default_rng(seed)
    solver = ALNS(rng)

    # [FIX v2] The installed alns version's add_*_operator logs op.__name__
    # UNCONDITIONALLY (before it even looks at an explicit name= kwarg), so
    # passing name= alone doesn't help -- functools.partial objects have no
    # __name__ at all and it still crashes. functools.partial supports
    # arbitrary attribute assignment, so set __name__ directly on each
    # partial before registering it; this satisfies the library's internal
    # logging regardless of whether/how it also uses name=.
    destroy_random = functools.partial(alns_random_removal, degree_of_destruction=degree_of_destruction)
    destroy_random.__name__ = 'alns_random_removal'
    destroy_string = functools.partial(alns_string_removal, customer_coords=customer_coords,
                                        max_string_removals=max_string_removals,
                                        max_string_size=max_string_size)
    destroy_string.__name__ = 'alns_string_removal'
    repair_greedy = functools.partial(alns_greedy_repair, customer_demands=customer_demands, capacity=capacity)
    repair_greedy.__name__ = 'alns_greedy_repair'

    solver.add_destroy_operator(destroy_random, name='alns_random_removal')
    solver.add_destroy_operator(destroy_string, name='alns_string_removal')
    solver.add_repair_operator(repair_greedy, name='alns_greedy_repair')

    select = RouletteWheel([25, 5, 1, 0], 0.8, 2, 1)
    accept = RecordToRecordTravel.autofit(state.objective(), 0.02, 0, num_iterations)
    stop = MaxIterations(num_iterations)

    result = solver.iterate(state, select, accept, stop)
    best = result.best_state

    exact_total = 0.0
    refined_clusters = []
    for route in best.routes:
        if not route:
            continue
        ids = [customer_ids[i] for i in route]
        d, _ = solve_tsp_cluster(ids, vrp_nodes, time_limit_seconds=tsp_time_limit)
        exact_total += d
        refined_clusters.append(route)

    return refined_clusters, exact_total
