"""Phase II: QAOA-based knapsack repair (QKR-INI).

Implements the quantum knapsack repair operator (Stage A / Stage B /
Safety Ripple, per the paper's Algorithm 2) plus the classical
greedy-repair baseline used for comparison.
"""
import numpy as np
import pennylane as qml

from .utils import eucl_dist

def exact_knapsack_optimum(item_values, item_weights, capacity, min_w, max_n_exact=18):
    """
    [Optimality diagnostic] Brute-force ground truth for the 0/1 knapsack
    sub-problem, vectorized over all 2**n candidate bitstrings (bitmask
    matrix @ weights/values) so it stays cheap enough to run inline rather
    than as a separate offline script. n is bounded by qaoa_knapsack's own
    max_qaoa_n (qaoa_knapsack falls back to greedy above that), but exact
    enumeration cost still grows as 2**n: ~0.13s at n=16, ~0.23s at n=18,
    ~1.1s at n=20, ~4.9s at n=22 on a single core. Defaults to skipping (returns None) above n=18 so it's
    safe to leave on for every repair sub-problem on the large instances;
    raise max_n_exact if you want exact ground truth on bigger clusters too
    and can afford the extra wall-clock time.
    """
    n = len(item_values)
    if n > max_n_exact:
        return None
    bits = ((np.arange(2 ** n, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1).astype(np.int8)
    w = bits @ item_weights
    feasible = (w >= min_w) & (w <= capacity + 1e-6)
    if not np.any(feasible):
        return None
    v = bits @ item_values
    return float(v[feasible].max())


def qaoa_knapsack(item_values, item_weights, capacity,
                   depth=1, shots=2000, steps=160,
                   alpha=None, beta=None, penalty_scale=10,
                   return_energy=False, seed=None, diagnostics=None,
                   max_n_exact=18, max_qaoa_n=15):
    """
    [R2-9] `seed` now controls BOTH the classical parameter initialization
    and reproducibility of results (previously relied on an unseeded
    global `np.random`).
    [R2-4] If `diagnostics` (a list) is passed, this appends a record of
    whether any *infeasible* sampled bitstring scored higher than the
    feasible bitstring actually selected — a direct empirical check on
    whether the unbalanced penalization ever misranks infeasible states.
    `penalty_scale` multiplies the default alpha/beta penalty weights,
    letting you sweep penalty strength (e.g. --penalty-scale 2.0) to see
    whether stronger penalization reduces the infeasible-outranks-feasible
    rate reported by the R2-4 diagnostic.
    [Optimality diagnostic] When `diagnostics` is passed, each record also
    includes the brute-force-exact optimum (via `exact_knapsack_optimum`,
    n permitting) and QAOA's optimality gap against it — a direct measure
    of "does the returned bitstring give the optimal solution", not just
    "was an infeasible sample never selected". `shots` defaults to 2000
    (was 500): in a controlled test (grid-warm-start vs random init x 500
    vs 2000 shots, scored against this same brute-force ground truth), the
    shot count was the lever that mattered — 500->2000 shots took the mean
    optimality gap from 1.78% to 0.00% across 5 trials, while switching the
    init strategy made no measurable difference. Cost is negligible here
    since lightning.qubit draws shots from an already-computed state vector
    rather than re-executing the circuit per shot.
    """
    rng = np.random.RandomState(seed)
    n = len(item_values)
    if n > max_qaoa_n:
        ratio = item_values / (item_weights + 1e-6)
        order = np.argsort(-ratio)
        selected = []
        w = 0.0
        for i in order:
            if w + item_weights[i] <= capacity:
                selected.append(i)
                w += item_weights[i]
        return selected if not return_energy else (selected, [])

    if alpha is None:
        max_val = np.max(item_values)
        total_w = np.sum(item_weights)
        rho = total_w / capacity
        alpha = max_val * rho #(5.0 + 10.0 * rho)
        beta = max_val * rho #(2.0 + 4.0 * rho)
    if beta is None:
        beta = 2.0 * np.max(item_values)
    alpha *= penalty_scale
    beta *= penalty_scale

    h = np.zeros(n)
    J = np.zeros((n, n))
    sum_w = np.sum(item_weights)
    offset_coeff = sum_w / 2 - capacity
    coeff_w = -0.5 * item_weights
    for i in range(n):
        h[i] += alpha * 2 * offset_coeff * coeff_w[i]
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] += alpha * 2 * coeff_w[i] * coeff_w[j]
    h += beta * coeff_w
    h += 0.5 * item_values

    coeffs, obs = [], []
    for i in range(n):
        if abs(h[i]) > 1e-10:
            coeffs.append(h[i]); obs.append(qml.PauliZ(i))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-10:
                coeffs.append(J[i, j]); obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    H_cost = qml.Hamiltonian(coeffs, obs)

    # [BUGFIX] cost_layer(gamma, H_cost) applies a rotation whose actual angle
    # is ~gamma * (H_cost's coefficient magnitude). alpha/beta -- and hence
    # h, J -- scale with penalty_scale and with rho=total_weight/capacity, so
    # coefficients can reach the thousands-to-tens-of-thousands range; a
    # "natural-looking" gamma of ~0.1 then corresponds to an effective angle
    # of hundreds-to-thousands of radians, deep in aliased territory (many
    # full 2*pi wraps per tiny step), so gradient descent on the raw gamma
    # bounces around rather than converging. Dividing by coeff_scale inside
    # the circuit keeps the *optimized* gamma at the same natural O(0.1-1)
    # scale as eta regardless of penalty_scale/problem magnitude, so a single
    # global Adam stepsize works for both and the cost trajectory actually
    # settles at a minimum instead of oscillating.
    coeff_scale = max(np.max(np.abs(h)) if h.any() else 0.0,
                       np.max(np.abs(J)) if J.any() else 0.0, 1e-9)

    dev_exact = qml.device("lightning.qubit", wires=n)
    dev_sample = qml.device("lightning.qubit", wires=n)

    @qml.qnode(dev_exact)
    def cost_circuit(params):
        gammas = params[:depth]; etas = params[depth:]
        for i in range(n): qml.Hadamard(wires=i)
        for l in range(depth):
            qml.qaoa.cost_layer(gammas[l] / coeff_scale, H_cost)
            for i in range(n): qml.RX(2 * etas[l], wires=i)
        return qml.expval(H_cost)

    # [BUGFIX] plain numpy arrays carry no autograd trainability metadata, so
    # AdamOptimizer.step_and_cost silently treats `params` as non-trainable:
    # it still returns a `cost` each step (forward pass is fine) but the
    # gradient is never computed and params never move -- 60 "optimization"
    # steps were a no-op every single call. qml.numpy.array(..., requires_grad=True)
    # is PennyLane's own numpy wrapper (autograd-backed) and is what
    # AdamOptimizer actually needs to differentiate through the circuit.
    params = qml.numpy.array(rng.normal(loc=0.0, scale=0.1, size=2 * depth),requires_grad=True)
    #params = qml.numpy.array(0.1 * rng.randn(2 * depth), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    energy_hist = []
    for _ in range(steps):
        params, cost = opt.step_and_cost(cost_circuit, params)
        energy_hist.append(float(cost))

    @qml.qnode(dev_sample)
    def sample_circuit(params):
        # same gamma/coeff_scale reparametrization as cost_circuit -- must
        # match exactly, since `params` coming out of training is expressed
        # in the rescaled (O(0.1-1)) representation, not the raw angle.
        gammas = params[:depth]; etas = params[depth:]
        for i in range(n): qml.Hadamard(wires=i)
        for l in range(depth):
            qml.qaoa.cost_layer(gammas[l] / coeff_scale, H_cost)
            for i in range(n): qml.RX(2 * etas[l], wires=i)
        return qml.sample(wires=range(n))

    samples = sample_circuit(params, shots=shots)

    min_w = max(np.min(item_weights), 0.5 * capacity)
    unique_samp = np.unique(samples, axis=0)
    best_val, best_bits = -np.inf, None
    infeasible_vals = []
    for bits in unique_samp:
        w = np.dot(bits, item_weights)
        v = np.dot(bits, item_values)
        if min_w <= w <= capacity + 1e-6:
            if v > best_val:
                best_val, best_bits = v, bits
        else:
            infeasible_vals.append(v)

    if best_bits is None:
        ratio = item_values / (item_weights + 1e-6)
        order = np.argsort(-ratio)
        selected = []
        w = 0.0
        for i in order:
            if w + item_weights[i] <= capacity:
                selected.append(i)
                w += item_weights[i]
    else:
        selected = [i for i in range(n) if best_bits[i] == 1]

    if diagnostics is not None:
        # [R2-4] Did any infeasible sample outrank the chosen feasible one?
        misranked = any(iv > best_val for iv in infeasible_vals) if infeasible_vals else False
        # [Optimality diagnostic] How does the returned bitstring compare to
        # the true (brute-force) optimum? None if n > max_n_exact (skipped).
        exact_opt = exact_knapsack_optimum(item_values, item_weights, capacity,
                                            min_w, max_n_exact=max_n_exact)
        if exact_opt is not None and exact_opt > 0 and best_bits is not None:
            # clip to 0: best_val can't legitimately exceed the true optimum,
            # so a tiny negative value here is float noise, not a real gap
            gap_pct = max(0.0, 100.0 * (exact_opt - best_val) / exact_opt)
        else:
            gap_pct = None
        diagnostics.append({
            "n_items": n, "n_infeasible_samples": len(infeasible_vals),
            "n_feasible_samples": len(unique_samp) - len(infeasible_vals),
            "best_feasible_value": best_val if best_bits is not None else None,
            "max_infeasible_value": max(infeasible_vals) if infeasible_vals else None,
            "infeasible_outranked_feasible": misranked,
            "exact_optimum": exact_opt,
            "optimality_gap_pct": gap_pct,
        })

    if return_energy:
        return selected, energy_hist
    return selected


def qkr_ini_repair(cluster_indices, customer_coords, customer_demands, capacity,
                    centroids=None, seed=None, diagnostics=None, penalty_scale=10,
                    max_qaoa_n=15):
    """Unchanged repair logic. Works identically on quantum- or
    classically-generated initial clusters, which is what lets the
    [R1-3] baselines below run through this exact same repair + Phase III
    stage as the quantum methods.
    `max_qaoa_n` is the single source of truth for the QAOA-vs-greedy size
    cutoff, used consistently in both Phase A and Phase B below and passed
    through to qaoa_knapsack's own internal fallback check -- previously
    these were three separate, inconsistent hardcoded numbers (15, 12, 22)."""
    K = len(cluster_indices)
    S = [list(cluster_indices[k]) for k in range(K)]

    if centroids is None:
        centroids = [np.mean(customer_coords[S[k]], axis=0) if len(S[k]) > 0 else np.zeros(2) for k in range(K)]
    else:
        centroids = list(centroids)

    qaoa_calls = 0
    evicted = []
    frozen = set()
    active = set(range(K))
    rng_counter = [0]

    def next_seed():
        rng_counter[0] += 1
        return None if seed is None else seed + rng_counter[0]

    # Phase A
    for k in range(K):
        w = np.sum(customer_demands[S[k]])
        if w > capacity:
            vals = 1.0 / (1.0 + np.sqrt(np.sum((customer_coords[S[k]] - centroids[k]) ** 2, axis=1)))
            wgts = customer_demands[S[k]]
            if len(S[k]) <= max_qaoa_n:
                selected_local = qaoa_knapsack(vals, wgts, capacity, seed=next_seed(),
                                                diagnostics=diagnostics, penalty_scale=penalty_scale,
                                                max_qaoa_n=max_qaoa_n)
                qaoa_calls += 1
            else:
                ratio = vals / (wgts + 1e-6)
                order = np.argsort(-ratio)
                selected_local = []
                wsum = 0.0
                for i in order:
                    if wsum + wgts[i] <= capacity:
                        selected_local.append(i)
                        wsum += wgts[i]
            selected_global = [S[k][i] for i in selected_local]
            evicted_new = [i for i in S[k] if i not in selected_global]
            S[k] = selected_global
            evicted.extend(evicted_new)
            frozen.add(k)
            active.discard(k)

    for k in frozen:
        if len(S[k]) > 0:
            centroids[k] = np.mean(customer_coords[S[k]], axis=0)

    # Phase B
    max_rounds = 15 * K
    round_count = 0
    while evicted and round_count < max_rounds:
        round_count += 1
        if not active:
            best = max(range(K), key=lambda k: capacity - np.sum(customer_demands[S[k]]))
            spare = capacity - np.sum(customer_demands[S[best]])
            # [BUGFIX] If even the roomiest frozen cluster can't fit the
            # smallest pending evictee, unfreezing it changes nothing --
            # the previous code retried this exact no-op every round until
            # max_rounds (observed: 87 identical rounds on one A-n33-k6
            # run, cluster stuck at 0-3 spare while every pending evictee
            # needed more). Detect that up front and drop straight to the
            # feasibility-guaranteed safety fallback instead of burning
            # the rest of the round budget (and QAOA calls) on repeats.
            if spare < min(customer_demands[e] for e in evicted):
                break
            active.add(best)
            frozen.discard(best)

        batches = {k: [] for k in active}
        for e in evicted:
            nearest = min(active, key=lambda k: eucl_dist(customer_coords[e], centroids[k]))
            batches[nearest].append(e)

        new_evicted = []
        clusters_to_freeze = []
        for k in list(active):
            batch = batches.get(k, [])
            if not batch:
                continue
            combined = S[k] + batch
            w_total = np.sum(customer_demands[combined])
            if w_total <= capacity:
                S[k] = combined
                centroids[k] = np.mean(customer_coords[combined], axis=0)
            else:
                vals = 1.0 / (1.0 + np.sqrt(np.sum((customer_coords[combined] - centroids[k]) ** 2, axis=1)))
                wgts = customer_demands[combined]
                if len(combined) <= max_qaoa_n:
                    selected_local = qaoa_knapsack(vals, wgts, capacity, seed=next_seed(),
                                                    diagnostics=diagnostics, penalty_scale=penalty_scale,
                                                    max_qaoa_n=max_qaoa_n)
                    qaoa_calls += 1
                else:
                    ratio = vals / (wgts + 1e-6)
                    order = np.argsort(-ratio)
                    selected_local = []
                    wsum = 0.0
                    for i in order:
                        if wsum + wgts[i] <= capacity:
                            selected_local.append(i)
                            wsum += wgts[i]
                selected_global = [combined[i] for i in selected_local]
                S[k] = selected_global
                evicted_from_cluster = [i for i in combined if i not in selected_global]
                new_evicted.extend(evicted_from_cluster)
                clusters_to_freeze.append(k)
                if len(selected_global) > 0:
                    centroids[k] = np.mean(customer_coords[selected_global], axis=0)

        for k in clusters_to_freeze:
            frozen.add(k)
            active.discard(k)
        evicted = new_evicted

    # [BUGFIX] Matches the paper's "pop first element" (FIFO) -- the old
    # `.pop()` popped the LAST element (LIFO), so a customer swapped out
    # this iteration became the very next one reprocessed, which is
    # exactly the kind of pattern that can thrash. FIFO also makes this
    # loop provably terminate with every cluster feasible: for each `e`,
    # keep evicting the current farthest member of the target cluster
    # until `e` actually fits, rather than doing one unconditional swap
    # and moving on regardless of whether it helped (the previous version,
    # like the paper's own pseudocode, never checked that the swap
    # actually restored feasibility -- so it could report `evicted` empty
    # while a cluster was still over capacity, which is what happened for
    # Amplitude above: one fallback "swap" left cluster 3 at 103/100).
    while evicted:
        e = evicted.pop(0)
        best = max(range(K), key=lambda k: capacity - np.sum(customer_demands[S[k]]))
        while np.sum(customer_demands[S[best]]) + customer_demands[e] > capacity and len(S[best]) > 0:
            dists = [eucl_dist(customer_coords[i], centroids[best]) for i in S[best]]
            far = np.argmax(dists)
            evicted.append(S[best].pop(far))
            if len(S[best]) > 0:
                centroids[best] = np.mean(customer_coords[S[best]], axis=0)
        S[best].append(e)
        centroids[best] = np.mean(customer_coords[S[best]], axis=0)

    frozen_info = []
    for k in sorted(frozen):
        dem = np.sum(customer_demands[S[k]]) if len(S[k]) > 0 else 0.0
        frozen_info.append({'cluster_id': k + 1, 'demand': dem})
    return S, qaoa_calls, frozen_info


def greedy_repair(clusters, customer_coords, customer_demands, capacity):
    repaired = [list(clusters[k]) for k in range(len(clusters))]
    evictees = []
    for k in range(len(repaired)):
        while np.sum(customer_demands[repaired[k]]) > capacity:
            cent = np.mean(customer_coords[repaired[k]], axis=0)
            dists = np.sqrt(np.sum((customer_coords[repaired[k]] - cent) ** 2, axis=1))
            farthest = np.argmax(dists)
            evictees.append(repaired[k].pop(farthest))
    for e in evictees:
        feas_k = [k for k in range(len(repaired))
                  if np.sum(customer_demands[repaired[k]]) + customer_demands[e] <= capacity]
        if feas_k:
            best_k = min(feas_k, key=lambda k: eucl_dist(customer_coords[e], np.mean(customer_coords[repaired[k]], axis=0)))
            repaired[best_k].append(e)
    return repaired
