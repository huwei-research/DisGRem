"""
graph.py – Random graph generation utilities
Ported from MATLAB: GenerateRandomGraph.m, generate_connected_graph.m

Supported topologies
--------------------
  random    – connected random geometric graph (default, varies per seed)
  ring      – cycle graph (deterministic)
  grid      – 2-D grid / mesh (deterministic)
  complete  – fully-connected averaging (deterministic)
  star      – hub-and-spoke (deterministic)
"""

from __future__ import annotations
import numpy as np
from scipy.sparse.csgraph import connected_components


def generate_random_graph(N: int, r: float):
    """
    Generate a connected random geometric graph.

    Nodes are placed uniformly in the 2-D unit square.
    Two nodes are connected if their Euclidean distance < r.
    The process repeats until the graph is connected.

    Parameters
    ----------
    N : number of nodes
    r : connection radius in (0, 1]

    Returns
    -------
    A : (N, N) ndarray – symmetric adjacency matrix
    W : (N, N) ndarray – Metropolis consensus matrix  W = I − η L
    """
    if not (0 < r <= 1):
        raise ValueError("r must be in (0, 1]")

    A = np.zeros((N, N))
    while True:
        A[:] = 0
        pos = np.random.rand(N, 2)

        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(pos[i] - pos[j]) < r:
                    A[i, j] = 1
                    A[j, i] = 1

        # connectivity check
        n_comp, _ = connected_components(A, directed=False)
        if n_comp == 1:
            break

    deg = A.sum(axis=1)
    d_max = deg.max()
    L = np.diag(deg) - A
    eta = 1.0 / d_max
    W = np.eye(N) - eta * L
    return A, W


def generate_connected_graph(N: int, r: float) -> dict:
    """
    Same as generate_random_graph but returns a dict with extra info.
    """
    A, W = generate_random_graph(N, r)
    deg = A.sum(axis=1)
    d_max = deg.max()
    L = np.diag(deg) - A
    eta = 1.0 / d_max
    return {
        "A": A,
        "W": W,
        "L": L,
        "eta": eta,
        "degVec": deg,
    }


def _metropolis_weights(A: np.ndarray) -> np.ndarray:
    """Compute Metropolis–Hastings mixing matrix from adjacency A.

    If the standard Metropolis W has any negative eigenvalue (e.g. even-length
    ring has eigenvalue −1, causing permanent oscillation), the "lazy" variant
    W_lazy = (I + W_metro) / 2 is returned instead.  This shifts all eigenvalues
    into [0, 1] while preserving double-stochasticity.
    For most graphs (random, grid, star) no negative eigenvalues arise and the
    standard weights are used (preserving the original spectral gap).
    """
    N = A.shape[0]
    deg = A.sum(axis=1)
    d_max = float(deg.max())
    if d_max == 0:
        return np.eye(N)
    L = np.diag(deg) - A
    W_metro = np.eye(N) - (1.0 / d_max) * L
    # Apply lazy correction only when negative eigenvalues are present
    min_eig = float(np.linalg.eigvalsh(W_metro).min())
    if min_eig < -1e-9:
        return 0.5 * (np.eye(N) + W_metro)
    return W_metro


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic topology generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_ring_graph(N: int):
    """
    Ring (cycle) graph: node i connected to (i-1) and (i+1) mod N.

    Returns
    -------
    A : (N, N) symmetric adjacency matrix
    W : (N, N) Metropolis consensus matrix
    """
    if N < 2:
        return np.zeros((N, N)), np.ones((N, N))
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[(i + 1) % N, i] = 1.0
    return A, _metropolis_weights(A)


def generate_star_graph(N: int):
    """
    Star (hub-and-spoke) graph: node 0 is the hub; all others connect only to 0.

    Returns
    -------
    A : (N, N) symmetric adjacency matrix
    W : (N, N) Metropolis consensus matrix
    """
    if N < 2:
        return np.zeros((N, N)), np.ones((N, N))
    A = np.zeros((N, N))
    for i in range(1, N):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    return A, _metropolis_weights(A)


def generate_grid_graph(N: int):
    """
    2-D grid (mesh) graph with approximately sqrt(N) columns.

    Nodes are indexed row-major.  Boundary nodes have degree 2 or 3.
    If the resulting graph is disconnected (last partial row), isolated
    components are stitched to node 0.

    Returns
    -------
    A : (N, N) symmetric adjacency matrix
    W : (N, N) Metropolis consensus matrix
    """
    cols = max(2, int(np.ceil(np.sqrt(N))))
    A = np.zeros((N, N))
    for i in range(N):
        _, c = divmod(i, cols)
        if c + 1 < cols and i + 1 < N:       # right neighbour
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        if i + cols < N:                       # bottom neighbour
            A[i, i + cols] = 1.0
            A[i + cols, i] = 1.0

    # Ensure connectivity
    n_comp, labels = connected_components(A, directed=False)
    if n_comp > 1:
        for comp_id in range(1, n_comp):
            orphan = int(np.where(labels == comp_id)[0][0])
            A[0, orphan] = 1.0
            A[orphan, 0] = 1.0

    return A, _metropolis_weights(A)


def generate_fully_connected_graph(N: int):
    """
    Complete (fully-connected) graph with uniform averaging matrix W = (1/N) 1 1^T.

    Returns
    -------
    A : (N, N) all-ones (off-diagonal)
    W : (N, N) = (1/N) * ones(N, N)
    """
    A = np.ones((N, N)) - np.eye(N)
    W = np.ones((N, N)) / float(N)
    return A, W


# ─────────────────────────────────────────────────────────────────────────────
#  Topology registry
# ─────────────────────────────────────────────────────────────────────────────

def get_topology_generators() -> dict:
    """
    Return a dict mapping topology name → callable(N) → (A, W).

    Available topologies
    --------------------
    random   – connected random geometric graph (radius scales with N)
    ring     – cycle graph
    grid     – 2-D grid
    complete – fully connected
    star     – hub-and-spoke (hub = node 0)
    """
    return {
        "random":   lambda N: generate_random_graph(N, max(0.35, min(0.9, 2.8 / N))),
        "ring":     generate_ring_graph,
        "grid":     generate_grid_graph,
        "complete": generate_fully_connected_graph,
        "star":     generate_star_graph,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral gap utility
# ─────────────────────────────────────────────────────────────────────────────

def spectral_gap(W: np.ndarray) -> float:
    """
    Return the spectral gap of the mixing matrix W.

    Spectral gap = 1 - |second largest eigenvalue| of W.
    A larger value indicates faster mixing.
    """
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
    if len(eigvals) < 2:
        return 1.0
    return float(1.0 - eigvals[1])


def get_W(W_spec, k: int) -> np.ndarray:
    """
    Return mixing matrix for iteration k.
    Supports static ndarray or callable W_spec(k).
    """
    if callable(W_spec):
        return W_spec(k)
    return W_spec
