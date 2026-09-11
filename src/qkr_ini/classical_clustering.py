"""[R1-3] Classical CVRP-specific clustering baselines, fed through the
same Phase II/III pipeline as the quantum methods for a fair comparison."""
import numpy as np
from sklearn.cluster import DBSCAN


def sweep_clustering(coords, demands, capacity, depot, n_clusters):
    """Gillett & Miller (1974) angular sweep -- the canonical classical
    cluster-first CVRP heuristic."""
    angles = np.arctan2(coords[:, 1] - depot[1], coords[:, 0] - depot[0])
    order = np.argsort(angles)
    clusters = [[] for _ in range(n_clusters)]
    loads = [0.0] * n_clusters
    k = 0
    for idx in order:
        if loads[k] + demands[idx] > capacity and k < n_clusters - 1:
            k += 1
        clusters[k].append(int(idx))
        loads[k] += demands[idx]
    return clusters


def demand_aware_kmedoids(coords, demands, capacity, n_clusters, seed=42, max_iter=50):
    """Capacitated K-medoids: greedy capacity-respecting assignment to the
    nearest medoid, followed by medoid re-selection."""
    rng = np.random.RandomState(seed)
    N = len(coords)
    medoid_idx = rng.choice(N, n_clusters, replace=False)
    clusters = [[] for _ in range(n_clusters)]
    for _ in range(max_iter):
        dists = np.linalg.norm(coords[:, None, :] - coords[medoid_idx][None, :, :], axis=2)
        order = np.argsort(np.min(dists, axis=1))
        clusters = [[] for _ in range(n_clusters)]
        loads = [0.0] * n_clusters
        for i in order:
            pref = np.argsort(dists[i])
            placed = False
            for k in pref:
                if loads[k] + demands[i] <= capacity:
                    clusters[k].append(int(i))
                    loads[k] += demands[i]
                    placed = True
                    break
            if not placed:
                k = int(np.argmin(loads))
                clusters[k].append(int(i))
                loads[k] += demands[i]
        new_medoid = medoid_idx.copy()
        for k, mem in enumerate(clusters):
            if not mem:
                continue
            sub = coords[mem]
            sub_d = np.sum(np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2), axis=1)
            new_medoid[k] = mem[int(np.argmin(sub_d))]
        if np.array_equal(new_medoid, medoid_idx):
            break
        medoid_idx = new_medoid
    return clusters


def dbscan_clustering(coords, n_clusters, min_samples=2, seed=42):
    """Spatial DBSCAN, capacity-agnostic by design (as in the classical
    literature); noise points are absorbed into the nearest cluster and
    the cluster count is rebalanced to n_clusters so it can be routed
    through the same fixed-K downstream pipeline."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min_samples).fit(coords)
    dists, _ = nn.kneighbors(coords)
    eps = np.median(dists[:, -1]) * 1.5
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_.copy()
    if -1 in labels:
        centroids = {lab: coords[labels == lab].mean(axis=0) for lab in set(labels) if lab != -1}
        for i in np.where(labels == -1)[0]:
            if centroids:
                best = min(centroids, key=lambda l: np.linalg.norm(coords[i] - centroids[l]))
                labels[i] = best
            else:
                labels[i] = 0
    uniq = sorted(set(labels))
    clusters = [np.where(labels == u)[0].tolist() for u in uniq]
    rng = np.random.RandomState(seed)
    while len(clusters) < n_clusters:
        biggest = max(range(len(clusters)), key=lambda k: len(clusters[k]))
        mem = clusters[biggest]
        if len(mem) < 2:
            break
        pts = coords[mem]
        order = np.argsort(pts[:, 0])
        half = max(1, len(mem) // 2)
        clusters[biggest] = [mem[i] for i in order[:half]]
        clusters.append([mem[i] for i in order[half:]])
    while len(clusters) > n_clusters:
        smallest = min(range(len(clusters)), key=lambda k: len(clusters[k]))
        mem = clusters.pop(smallest)
        if not clusters:
            clusters = [mem]
            break
        cent = [np.mean(coords[c], axis=0) if c else np.zeros(2) for c in clusters]
        tgt = min(range(len(clusters)), key=lambda k: np.linalg.norm(coords[mem].mean(axis=0) - cent[k]))
        clusters[tgt].extend(mem)
    return clusters
