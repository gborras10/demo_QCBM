import numpy as np
import matplotlib.pyplot as plt

def minimize_with_cost_history(
    cost_fn,
    *,
    x0,
    minimize_fn,
    method,
    options,
):
    """
    Minimize a cost function while tracking the optimization history.
    This function wraps a cost function and optimization routine to capture the cost
    value at each iteration, building a history of costs throughout the minimization process.
    Parameters
    ----------
    cost_fn : callable
        The cost function to minimize. Should accept an array-like input and return a scalar float value.
    x0 : array_like
        Initial guess for the optimization parameters.
    minimize_fn : callable
        The minimization function to use (e.g., scipy.optimize.minimize).
        Should have the signature: minimize_fn(fun, x0, method, callback, options).
    method : str
        Optimization method to pass to minimize_fn (e.g., 'BFGS', 'L-BFGS-B', 'Nelder-Mead').
    options : dict
        Dictionary of solver options to pass to minimize_fn.
    Returns
    -------
    tuple
        A tuple containing:
        - res : OptimizeResult
            The optimization result object returned by minimize_fn.
        - cost_history : np.ndarray
            1D array of float values representing the cost function evaluation at each iteration.
            Array has dtype float64.
    Notes
    -----
    The cost history is built by wrapping the cost function to capture intermediate values
    during the optimization callback. The first iteration value is not included in the history.
    Examples
    --------
    >>> from scipy.optimize import minimize
    >>> def quadratic(x):
    ...     return (x - 3)**2
    >>> x0 = [0.0]
    >>> result, history = minimize_with_cost_history(
    ...     quadratic,
    ...     x0=x0,
    ...     minimize_fn=minimize,
    ...     method='BFGS',
    ...     options={}
    ... )
    """
    cost_history: list[float] = []
    last_val: float | None = None

    def wrapped(x):
        nonlocal last_val
        last_val = float(cost_fn(x))
        return last_val

    def callback(xk):
        if last_val is not None:
            cost_history.append(last_val)

    res = minimize_fn(
        wrapped,
        x0=x0,
        method=method,
        callback=callback,
        options=options,
    )

    return res, np.asarray(cost_history, dtype=float)