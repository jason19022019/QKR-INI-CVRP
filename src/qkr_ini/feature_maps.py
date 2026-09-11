"""Quantum feature maps (encodings) and quantum-kernel matrix construction.

Seven encodings are provided: Amplitude, Angle, IQP, Z-Map, ZZ-Map,
U3-Single, U3-TwoAngle. `compute_kernel_matrix` dispatches between the
exact O(N^2) kernel and the [R1-1] Nystrom low-rank approximation
depending on instance size.
"""
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

num_features = 2
n_qubits_amp = int(np.ceil(np.log2(num_features)))
n_qubits = max(3, n_qubits_amp)
dev_kernel = qml.device("lightning.qubit", wires=n_qubits)


def z_map(x):
    for i in range(num_features):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)


def zz_map(x):
    z_map(x)
    for i in range(num_features - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])


def iqp_map(x):
    qml.IQPEmbedding(x, wires=range(num_features), n_repeats=2)


def angle_map(x):
    qml.AngleEmbedding(x, wires=range(num_features), rotation='Y')


def amplitude_map(x):
    vec = np.zeros(2 ** n_qubits)
    vec[:num_features] = x
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    qml.AmplitudeEmbedding(vec, wires=range(n_qubits), normalize=True)


def u3_single_map(x):
    if len(x) >= 3:
        qml.U3(x[0], x[1], x[2], wires=0)
    elif len(x) == 2:
        qml.U3(x[0], x[1], 0.0, wires=0)
    else:
        qml.U3(x[0], 0.0, 0.0, wires=0)


def u3_two_angle_map(x):
    if len(x) >= 2:
        alpha, beta = x[0], x[1]
    elif len(x) == 1:
        alpha, beta = x[0], 0.0
    else:
        alpha, beta = 0.0, 0.0
    theta = (alpha + 1.0) * np.pi
    phi = (beta + 1.0) * np.pi
    qml.U3(theta, phi, 0.0, wires=0)


@qml.qnode(dev_kernel)
def compute_kernel(x1, x2, method):
    if method == 'Amplitude': amplitude_map(x1)
    elif method == 'Angle': angle_map(x1)
    elif method == 'IQP': iqp_map(x1)
    elif method == 'Z-Map': z_map(x1)
    elif method == 'ZZ-Map': zz_map(x1)
    elif method == 'U3-Single': u3_single_map(x1)
    elif method == 'U3-TwoAngle': u3_two_angle_map(x1)

    if method == 'Amplitude': qml.adjoint(amplitude_map)(x2)
    elif method == 'Angle': qml.adjoint(angle_map)(x2)
    elif method == 'IQP': qml.adjoint(iqp_map)(x2)
    elif method == 'Z-Map': qml.adjoint(z_map)(x2)
    elif method == 'ZZ-Map': qml.adjoint(zz_map)(x2)
    elif method == 'U3-Single': qml.adjoint(u3_single_map)(x2)
    elif method == 'U3-TwoAngle': qml.adjoint(u3_two_angle_map)(x2)

    return qml.probs(wires=range(n_qubits))


def _kernel_entry(x1, x2, method):
    probs = compute_kernel(x1[:num_features], x2[:num_features], method)
    return float(np.clip(probs[0], 0.0, 1.0))


def compute_kernel_matrix_exact(data, method):
    N = len(data)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            K[i, j] = K[j, i] = 1.0 if i == j else _kernel_entry(data[i], data[j], method)
    return K


def compute_kernel_matrix_nystrom(data, method, n_landmarks, seed=0):
    """
    [R1-1] Nystrom low-rank approximation of the quantum kernel matrix.
    Reduces quantum-circuit evaluations from O(N^2) to O(N*m + m^2) for
    m landmarks, making Phase I tractable on instances well beyond the
    101-customer ceiling flagged by Reviewer #1. Landmarks are chosen
    uniformly at random; approximation error vs. landmark count can be
    validated with `report_nystrom_quality` (run via --validate-nystrom).
    """
    rng = np.random.RandomState(seed)
    N = len(data)
    m = min(n_landmarks, N)
    landmarks = rng.choice(N, size=m, replace=False)
    C = np.zeros((N, m))
    for i in range(N):
        for li, l in enumerate(landmarks):
            C[i, li] = 1.0 if i == l else _kernel_entry(data[i], data[l], method)
    W = C[landmarks, :]
    W_pinv = np.linalg.pinv(W)
    K_approx = C @ W_pinv @ C.T
    np.fill_diagonal(K_approx, 1.0)
    return np.clip(K_approx, 0.0, 1.0), landmarks


def compute_kernel_matrix(data, method, large_threshold=150, n_landmarks=60, seed=0):
    """[R1-1] Dispatches to the exact or Nystrom-approximated kernel
    depending on instance size, and reports which path + circuit-call
    count was used (for the paper's scalability discussion)."""
    N = len(data)
    if N > large_threshold:
        K, landmarks = compute_kernel_matrix_nystrom(data, method, n_landmarks, seed)
        calls = N * len(landmarks)
        print(f"    [Nystrom] N={N} > {large_threshold}: used {len(landmarks)} "
              f"landmarks -> {calls} circuit evals (vs {N*(N-1)//2} exact).")
        return K
    else:
        K = compute_kernel_matrix_exact(data, method)
        print(f"    [Exact] N={N}: {N*(N-1)//2} circuit evals.")
        return K


def report_nystrom_quality(data, method, m_values, seed=0, large_threshold=150,
                            save_path="nystrom_convergence.png"):
    """
    [R1-1 diagnostic] Validates the Nystrom approximation against the exact
    quantum kernel, computed on a held-out subset (N <= large_threshold,
    where exact computation is still tractable), sweeping landmark count m
    and reporting relative Frobenius-norm error for each. Saves a
    convergence plot (error vs. m). Meant to be run once for the paper's
    scalability subsection via --validate-nystrom, not on every pipeline run.
    """
    N = min(len(data), large_threshold)
    sub = data[:N]
    print(f"[Nystrom validation] computing exact kernel on N={N} points ({method})...")
    K_exact = compute_kernel_matrix_exact(sub, method)
    errors = []
    for m in m_values:
        K_approx, _ = compute_kernel_matrix_nystrom(sub, method, m, seed)
        err = np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact)
        errors.append(err)
        print(f"  m={m}: relative Frobenius error = {err * 100:.3f}%")

    plt.figure(figsize=(6, 4))
    plt.plot(m_values, [e * 100 for e in errors], marker='o')
    plt.xlabel("Number of Nystrom landmarks (m)")
    plt.ylabel("Relative Frobenius error (%)")
    plt.title(f"Nystrom Kernel Approximation Convergence\n({method} encoding, N={N})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Nystrom validation] convergence plot saved to {save_path}")
    return m_values, errors
