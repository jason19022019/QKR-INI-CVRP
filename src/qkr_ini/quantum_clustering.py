"""Phase I: quantum kernel k-means clustering."""
import numpy as np
import matplotlib.pyplot as plt


def quantum_kernel_kmeans(K, n_clusters, max_iters=250, random_state=42, track_history=True):
    """
    Kernel k-means (Algorithm Phase I, Step 2). Centers are represented
    implicitly via the closed-form kernel-trick distance to each cluster's
    feature-space mean (standard practice, since a kernel-space mean
    generally has no explicit data-point pre-image to literally
    "recompute" as in the pseudocode's medoid-style wording) -- this is
    mathematically equivalent to, and a superset of, distance-to-medoid.

    [Convergence tracking] If `track_history=True`, also records, at every
    sweep, (a) the kernel k-means objective -- sum over all points of the
    squared kernel distance to their assigned cluster, i.e. the exact
    quantity `d2` below being minimized -- and (b) the number of points
    that changed label that sweep. This is what `plot_phase1_convergence`
    below plots, and is what the `Until labels stabilize or maximum
    iterations reached` stopping rule is actually driven by.
    """
    np.random.seed(random_state)
    N = K.shape[0]

    def kernel_dist_sq(i, j):
        return 2.0 - 2.0 * K[i, j]

    def kernel_dist_to_set_sq(i, S):
        if len(S) == 0:
            return np.inf
        return np.min([kernel_dist_sq(i, j) for j in S])

    center_indices = [np.random.randint(0, N)]
    for _ in range(1, n_clusters):
        dist_sq = np.array([kernel_dist_to_set_sq(i, center_indices) for i in range(N)])
        probs = dist_sq / np.sum(dist_sq)
        center_indices.append(np.random.choice(N, p=probs))
    labels = np.zeros(N, dtype=int)
    for i in range(N):
        labels[i] = np.argmin([kernel_dist_sq(i, c) for c in center_indices])

    history = {"objective": [], "n_changed": []}
    n_iters_run = 0
    for _ in range(max_iters):
        old_labels = labels.copy()
        sweep_objective = 0.0
        for i in range(N):
            best_cluster, best_dist = -1, np.inf
            for k in range(n_clusters):
                pts = np.where(labels == k)[0]
                if len(pts) == 0:
                    continue
                d2 = (K[i, i] - (2.0 / len(pts)) * np.sum(K[i, pts])
                      + (1.0 / len(pts) ** 2) * np.sum(K[np.ix_(pts, pts)]))
                if d2 < best_dist:
                    best_dist, best_cluster = d2, k
            labels[i] = best_cluster
            sweep_objective += best_dist
        n_changed = int(np.sum(old_labels != labels))
        n_iters_run += 1
        if track_history:
            history["objective"].append(sweep_objective)
            history["n_changed"].append(n_changed)
        if np.array_equal(old_labels, labels):
            break
    history["n_iters"] = n_iters_run
    history["converged"] = bool(n_iters_run < max_iters)
    if track_history:
        return labels, history
    return labels


def plot_phase1_convergence(phase_one_history, methods, save_path="figure_phase1_convergence.pdf"):
    """
    [Convergence check requested] Renders, for every quantum feature map
    run in Phase I, the kernel k-means convergence trace: the clustering
    objective (sum of squared kernel distances of every point to its
    assigned cluster) and the number of reassigned labels, sweep by sweep,
    up to the sweep where `labels stabilize or maximum iterations reached`
    (Algorithm Phase I, Step 2 stopping rule). A monotonically
    non-increasing objective curve that flattens out, together with
    n_changed hitting 0, is the empirical evidence that Phase I actually
    converged rather than being cut off by max_iters.
    """
    n_methods = len(methods)
    ncols = min(4, n_methods)
    nrows = int(np.ceil(n_methods / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for idx, m in enumerate(methods):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        hist = phase_one_history[m]
        iters = np.arange(1, len(hist["objective"]) + 1)
        ax.plot(iters, hist["objective"], color='#0072B2', marker='o', markersize=3, label='Objective')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Kernel k-means objective", color='#0072B2')
        ax.tick_params(axis='y', labelcolor='#0072B2')
        ax2 = ax.twinx()
        ax2.plot(iters, hist["n_changed"], color='#D55E00', marker='s', markersize=3,
                  linestyle='--', label='Labels changed')
        ax2.set_ylabel("# labels changed", color='#D55E00')
        ax2.tick_params(axis='y', labelcolor='#D55E00')
        status = "converged" if hist["converged"] else "hit max_iters"
        ax.set_title(f"{m}\n({hist['n_iters']} iters, {status})", fontsize=10)
        ax.grid(alpha=0.3)
    for idx in range(n_methods, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis('off')
    fig.suptitle("Phase I: Quantum Kernel $k$-Means Convergence", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Phase I] Convergence figure saved to {save_path}")
