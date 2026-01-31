from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Type definitions
Order = Literal["time_major", "price_major"]


@dataclass(frozen=True)
class JointQcbmTarget:
    """
    Joint target distribution for QCBM training over (time, price-bin).

    Attributes
    ----------
    p_tg : np.ndarray
        Flattened joint probability vector of length M * N.
    M, N : int
        Original sizes: M time steps, N price bins.
    m, n : int
        Number of qubits for time and price registers: M = 2**m, N = 2**n.
    order : {"time_major", "price_major"}
        Flattening convention.
        - time_major: index x = i * N + j  (time bits first, then price bits)
        - price_major: index x = j * M + i (price bits first, then time bits)
    time_weights : np.ndarray
        The weights pi_i used for P(t_i). Length M, sums to 1.
    """

    p_tg: np.ndarray
    M: int
    N: int
    m: int
    n: int
    order: Order
    time_weights: np.ndarray


def simulate_GBM_dynamics(
    S0: float,
    mu: float,
    sigma: float,
    t: np.ndarray,
    Z: np.ndarray,   # shape (N_paths, M)
) -> list[np.ndarray]:
    """
    Simulate marginal samples of a Geometric Brownian Motion at given time points.

    Parameters
    ----------
    S0 : float
        Initial underlying price at time t = 0.
    mu : float
        Drift parameter of the GBM.
    sigma : float
        Volatility of the GBM.
    t : np.ndarray
        Time grid of length M+1 with t[0] = 0 and t[i] the i-th exposure date.
    Z : np.ndarray
        Standard normal samples with shape (N_paths, M). Column i-1 is used
        to generate samples at time t_i.

    Returns
    -------
    list[np.ndarray]
        List of length M. The i-th element is an array of shape (N_paths,)
        containing samples of S(t_i).

    Example
    -------
    >>> import numpy as np
    >>> S0, mu, sigma = 5.0, 0.02, 0.25
    >>> t = np.linspace(0.0, 0.5, 5)   # M = 4 time steps
    >>> N_paths = 100_000
    >>> Z = np.random.standard_normal(size=(N_paths, 4))
    >>> S_by_time = simulate_GBM_dynamics(S0, mu, sigma, t, Z)
    >>> len(S_by_time)
    4
    >>> S_by_time[0]
    array([4.92017382, 5.58214855, 4.52283849, ..., 5.67746454, 4.53352251,
       5.00398244], shape=(100000,))
    """
    if Z.shape[1] != len(t) - 1:
        raise ValueError("Z shape and time grid t are inconsistent.")

    M = Z.shape[1]
    S_list = []
    for i in range(1, M + 1):
        ti = t[i]
        # Marginal GBM sampling at time t_i (no pathwise dynamics)
        Si = S0 * np.exp(
            (mu - 0.5 * sigma**2) * ti
            + sigma * np.sqrt(ti) * Z[:, i - 1]
        )
        S_list.append(Si)
    return S_list

def price_grid_from_samples(
    S_samples_by_time: list[np.ndarray],
    n: int,
    n_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a global uniform price grid with N = 2^n points from GBM samples.

    The grid range [s0, sN] is defined using the mean and standard deviation
    of the terminal distribution S(T), truncated to n_sigma standard deviations.

    Parameters
    ----------
    S_samples_by_time : list[np.ndarray]
        Samples of S(t_i) for each time step; only S(T) is used.
    n : int
        Number of qubits for the price register (N = 2^n grid points).
    n_sigma : float, optional
        Truncation width in standard deviations (default is 4.0).

    Returns
    -------
    edges : np.ndarray
        Array of length N+1 defining the bin edges of the price grid.
    s_mid : np.ndarray
        Array of length N with the representative price of each bin.

    Example
    -------
    >>> edges, s_mid = price_grid_from_samples(S_by_time, n=6)
    >>> len(s_mid)
    64
    """
    N = 2**n
    # use terminal time T (widest distribution) to define a global price grid
    X = S_samples_by_time[-1] 
    muhat = float(X.mean())
    sighat = float(X.std(ddof=1))
    s0 = max(muhat - n_sigma * sighat, 0.0)
    sN = muhat + n_sigma * sighat

    edges = np.linspace(s0, sN, N + 1)
    s_mid = 0.5 * (edges[:-1] + edges[1:])

    return edges, s_mid

def discrete_probs_from_samples(
    S: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Build a discrete probability distribution from continuous samples
    by histogramming over a fixed price grid and renormalizing.

    Parameters
    ----------
    S : np.ndarray
        Monte Carlo samples of the underlying price at a fixed time t_i.
    edges : np.ndarray
        Bin edges defining the price grid [s0, ..., sN].

    Returns
    -------
    p : np.ndarray
        Discrete probability vector of length len(edges)-1 such that
        p[j] ≈ P(S in bin j | S in [s0, sN]) and sum(p) = 1.

    Example
    -------
    >>> import numpy as np
    >>> S = np.array([1.2, 1.7, 2.1, 2.9])
    >>> edges = np.array([1.0, 2.0, 3.0])
    >>> p = discrete_probs_from_samples(S, edges)
    >>> p
    array([0.5, 0.5])
    """
    counts, _ = np.histogram(S, bins=edges)
    in_range = counts.sum()
    if in_range == 0:
        raise ValueError("No samples in range; widen [s0, sN].")
    return counts / in_range

def _check_and_normalize_prob_vector(p: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Validate and normalize a probability vector to sum to 1 within tolerance."""
    p = np.asarray(p, dtype=float).ravel()
    if np.any(p < -tol):
        raise ValueError("Probability vector has negative entries (beyond tolerance).")
    p = np.clip(p, 0.0, None) # remove small negatives
    prob_sum = float(p.sum())
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise ValueError("Probability vector has non-finite or non-positive sum.")
    if abs(prob_sum - 1.0) > tol:
        p = p / prob_sum
    return p


def build_joint_target_from_P_bin(
    P_bin: np.ndarray,
    *,
    order: Order = "time_major",
) -> JointQcbmTarget:
    r"""
    Build the joint target distribution used to train a QCBM, following:

    p_{tg}(i,j) = P(t_i, s_j) = P(s_j|t_i) · P(t_i)

    where P(s_j|t_i) is given by `P_bin[i, j]` (row-stochastic), and P(t_i) is
    uniform.

    Parameters
    ----------
    P_bin : np.ndarray, shape (M, N)
        Conditional probabilities per time: P(s_bin=j | t_i).
        Each row is renormalized internally to sum to 1.
    order : {"time_major", "price_major"}
        Flattening convention to map (i,j) -> x.
        - "time_major": x = i*N + j  (time register first, then price register)
        - "price_major": x = j*M + i

    Returns
    -------
    JointQcbmTarget
        Contains flattened p_tg and metadata (M, N, m, n, etc.).
    """
    P_bin = np.asarray(P_bin, dtype=float)
    if P_bin.ndim != 2:
        raise ValueError("P_bin must be a 2D array of shape (M, N).")

    M, N = P_bin.shape

    # Sanitize/renormalize each conditional row P(s|t_i)
    P = np.clip(P_bin, 0.0, None)
    row_sums = P.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        bad = np.where(row_sums.ravel() <= 0.0)[0]
        raise ValueError(f"At least one row in P_bin has zero mass. Bad rows: {bad.tolist()}")
    P = P / row_sums

    # Define P(t_i) = pi_i (uniform distribution over time steps)
    w = np.full(M, 1.0 / M, dtype=float)

    # Require sizes compatible with qubit registers
    if (M & (M - 1)) != 0 or (N & (N - 1)) != 0:
        raise ValueError("M and N must be powers of two.")

    # Joint p(t_i, s_j) = w_i * P(s_j|t_i)
    joint = w[:, None] * P  # shape (M, N)

    # Flatten in the chosen order
    if order == "time_major":
        p_tg = joint.reshape(M * N)
    elif order == "price_major":
        p_tg = joint.T.reshape(M * N)
    else:
        raise ValueError("order must be 'time_major' or 'price_major'.")

    p_tg = _check_and_normalize_prob_vector(p_tg)

    # Qubit counts (safe because M and N are powers of two)
    m = int(np.log2(M))
    n = int(np.log2(N))

    return JointQcbmTarget(
        p_tg=p_tg,
        M=M,
        N=N,
        m=m,
        n=n,
        order=order,
        time_weights=w,
    )