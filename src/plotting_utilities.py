import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_GBM_dynamics(
    S_by_time,
    t,
    S0,
    r,
    sigma,
    T,
    n_paths,
    n_plot=1000,
    figsize=(12, 7),
    colormap='viridis',
    alpha=0.6,
    linewidth=1.2
):
    """
    Plot sample paths from a Geometric Brownian Motion simulation.
    
    Creates a professional visualization of simulated GBM trajectories with
    customizable styling, including individual path colors, reference lines,
    and comprehensive parameter annotations in the title.
    
    Parameters
    ----------
    S_by_time : list of np.ndarray
        List of arrays containing simulated prices at each time step.
        Each element S_by_time[i] contains all path values at time t[i].
        Shape: [M time steps][n_paths].
    t : np.ndarray
        Time grid array of shape (M+1,) including t=0.
        Used for x-axis plotting (excluding initial time).
    S0 : float
        Initial underlying price. Used for reference line annotation.
    r : float
        Drift rate (expected return) of the GBM process.
        Displayed as percentage in the title.
    sigma : float
        Volatility (diffusion coefficient) of the GBM process.
        Displayed as percentage in the title.
    T : float
        Time horizon in years. Displayed in the title.
    n_paths : int
        Total number of simulated paths. Used for annotation to show
        the fraction of paths displayed.
    n_plot : int, optional
        Number of paths to plot from the available simulations.
        Default is 1000. If n_plot > n_paths, all paths are plotted.
    figsize : tuple of float, optional
        Figure dimensions (width, height) in inches. Default is (12, 7).
    colormap : str, optional
        Matplotlib colormap name for path coloring. Default is 'viridis'.
    alpha : float, optional
        Transparency level for individual paths (0=transparent, 1=opaque).
        Default is 0.6.
    linewidth : float, optional
        Line width for individual paths in points. Default is 1.2.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot elements.
    
    Notes
    -----
    The function applies professional styling including:
    - Light gray background (#fafafa) for improved readability
    - Color-coded paths using the specified colormap
    - Horizontal reference line at the initial price S0
    - Grid with subtle styling for easier value reading
    - Comprehensive title with all simulation parameters
    - Customized spines, ticks, and legend appearance
    
    The plot displays trajectories starting from t[1] onwards, excluding
    the initial time point t[0]=0.
    
    Examples
    --------
    >>> S_by_time = simulate_GBM_dynamics(S0=100, mu=0.05, sigma=0.2, t=t, Z=Z)
    >>> fig, ax = plot_gbm_sample_paths(
    ...     S_by_time, t, S0=100, r=0.05, sigma=0.2, T=1.0,
    ...     n_paths=10000, n_plot=500
    ... )
    >>> plt.savefig('gbm_paths.png', dpi=300, bbox_inches='tight')
    """
    # Prepare data for plotting
    n_plot_actual = min(n_plot, n_paths)
    S_plot = np.vstack([s[:n_plot_actual] for s in S_by_time])
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#fafafa')
    
    # Generate colors from colormap
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_plot_actual))
    
    # Plot individual paths
    for k in range(n_plot_actual):
        ax.plot(t[1:], S_plot[:, k], color=colors[k], alpha=alpha, 
                linewidth=linewidth, zorder=2)
    
    # Reference line at initial price
    ax.axhline(y=S0, color='#252525', linewidth=2, linestyle='--', 
               alpha=0.7, label=f'S₀ = {S0:.2f}', zorder=3)
    
    # Axis labels
    ax.set_xlabel('Time (years)', fontsize=13, fontweight='600', color='#1a1a1a')
    ax.set_ylabel('Underlying Price S(t)', fontsize=13, fontweight='600', 
                  color='#1a1a1a')
    
    # Title with simulation parameters
    ax.set_title(
        f'Geometric Brownian Motion: Sample Paths\n'
        f'S₀ = {S0} | μ = {r:.2%} | σ = {sigma:.2%} | T = {T} year | '
        f'{n_plot_actual}/{n_paths:,} paths',
        fontsize=15, fontweight='bold', pad=20, color='#1a1a1a'
    )
    
    # Grid styling
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, 
            color='#bdbdbd', zorder=0)
    ax.set_axisbelow(True)
    
    # Legend styling
    legend = ax.legend(loc='best', frameon=True, fancybox=True, 
                      shadow=False, fontsize=11, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#999')
    
    # Spines styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#999')
        spine.set_linewidth(1.5)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#333',
                   width=1.2, length=6)
    
    plt.tight_layout()
    
    return fig, ax

def plot_histogram_comparison(
    samples,
    edges,
    n_fine_bins=1000,
    figsize=(10, 3.5),
    raw_color="steelblue",
    discrete_color="slategray",
    alpha=0.8
):
    """
    Plot side-by-side comparison of raw and discretized histograms.
    
    Creates a two-panel visualization comparing the original fine-grained
    distribution of samples against the discretized version using specified
    bin edges. This is useful for visualizing the information loss inherent
    in discretization processes.
    
    Parameters
    ----------
    samples : np.ndarray
        1D array of continuous sample values to be histogrammed.
        Typically represents prices, returns, or other continuous variables.
    edges : np.ndarray
        Bin edges for the discretized histogram. Should be a 1D array of
        length (N+1) where N is the number of discrete bins.
        The bins are defined as [edges[i], edges[i+1]).
    n_fine_bins : int, optional
        Number of bins to use for the raw (fine-grained) histogram.
        Default is 1000. Higher values show more detail but may be noisier.
    figsize : tuple of float, optional
        Figure dimensions (width, height) in inches. Default is (10, 3.5).
    raw_color : str, optional
        Color for the raw histogram bars. Default is "steelblue".
    discrete_color : str, optional
        Color for the discretized histogram bars. Default is "slategray".
    alpha : float, optional
        Transparency level for histogram bars (0=transparent, 1=opaque).
        Default is 0.8.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing both subplots.
    axes : np.ndarray of matplotlib.axes.Axes
        Array of two axes objects [raw_axis, discrete_axis].
    
    Notes
    -----
    The function creates two subplots:
    - Left panel: Raw samples with many bins, showing the detailed distribution
    - Right panel: Discretized samples using provided edges, showing the coarse
      approximation used in quantum algorithms
    
    The discretized histogram uses bar plots with variable bin widths, which
    correctly represents non-uniform bin spacing if present in the edges array.
    
    Grid lines are added with low alpha for easier value reading without
    cluttering the visualization.
    
    Examples
    --------
    >>> samples = np.random.normal(100, 15, 10000)
    >>> edges = np.linspace(50, 150, 11)  # 10 bins
    >>> fig, axes = plot_histogram_comparison(samples, edges)
    >>> plt.show()
    
    >>> # For high-resolution comparison
    >>> fig, axes = plot_histogram_comparison(
    ...     samples, edges, n_fine_bins=2000, figsize=(12, 4)
    ... )
    >>> plt.savefig('histogram_comparison.png', dpi=300)
    """
    N_bins = len(edges) - 1
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Raw histogram (fine bins)
    axes[0].hist(samples, bins=n_fine_bins, color=raw_color, alpha=alpha)
    axes[0].set_title("Raw samples at T (fine bins)")
    axes[0].set_xlabel("S(T)")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.3)
    
    # Discretized histogram using the provided edges
    counts, _ = np.histogram(samples, bins=edges)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]
    
    axes[1].bar(
        bin_centers, counts, width=bin_widths,
        color=discrete_color, alpha=alpha, align="center"
    )
    axes[1].set_title(f"Discretized bins (N={N_bins})")
    axes[1].set_xlabel("S(T) bin")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, axes

def plot_training_diagnostics(
    *,
    target: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    cost_history: np.ndarray,
    best_so_far: np.ndarray | None = None,
    best_idx: np.ndarray | None = None,
    labels: list[str] | np.ndarray | None = None,
    x_values: np.ndarray | None = None,
    grid_info: dict | None = None,
    xlabel: str | None = None,
    ylabel: str = "Probability",
    title_before: str | None = None,
    title_after: str | None = None,
    cost_xlabel: str = "Optimization Step",
    cost_ylabel: str = "Rescaled Cost Function",
    bar_width: float = 0.85,
    figsize_dist: tuple[float, float] = (14, 5),
    figsize_cost: tuple[float, float] = (10, 5),
    max_states: int | None = None,
    cost_log_x: bool = False,
    cost_log_y: bool = True,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Plot training diagnostics: distributions comparison and cost evolution.
    """
    target = np.asarray(target, dtype=float).ravel()
    before = np.asarray(before, dtype=float).ravel()
    after = np.asarray(after, dtype=float).ravel()
    cost_history = np.asarray(cost_history, dtype=float).ravel()

    dim = target.shape[0]
    if before.shape[0] != dim or after.shape[0] != dim:
        raise ValueError(
            f"Incompatible dimensions: target({dim}), before({before.shape[0]}), after({after.shape[0]})."
        )
    if cost_history.size == 0:
        raise ValueError("cost_history must be non-empty.")

    if best_so_far is None:
        best_so_far = np.minimum.accumulate(cost_history)
    else:
        best_so_far = np.asarray(best_so_far, dtype=float).ravel()
        if best_so_far.shape != cost_history.shape:
            raise ValueError("best_so_far must have the same shape as cost_history.")

    if best_idx is None:
        tol = 1e-15
        improved = np.r_[True, best_so_far[1:] < best_so_far[:-1] - tol]
        best_idx = np.flatnonzero(improved)
    else:
        best_idx = np.asarray(best_idx, dtype=int).ravel()

    # ============================================================
    # Scenario detection: Price-only vs Time+Price
    # ============================================================
    if grid_info is not None and "s_mid" in grid_info:
        scenario = "price_only"
        s_mid = np.asarray(grid_info["s_mid"], dtype=float)
        if s_mid.shape[0] != dim:
            raise ValueError(f"grid_info['s_mid'] must have length {dim}; got {s_mid.shape[0]}.")
        use_price_values = True
    else:
        scenario = "time_price"
        use_price_values = False

    if xlabel is None:
        xlabel = "Underlying Price" if use_price_values else r"Computational basis state"

    if title_before is None:
        title_before = "S(T) Distribution - Before Training" if use_price_values else "Before Training"

    if title_after is None:
        title_after = "S(T) Distribution - After Training" if use_price_values else "After Training"

    if max_states is not None and dim > int(max_states):
        dim_plot = int(max_states)
        sl = slice(0, dim_plot)
    else:
        dim_plot = dim
        sl = slice(None)

    if use_price_values:
        x = s_mid[sl]
        ticklabels = [f"{val:.3g}" for val in x]
        use_x_values = True
    elif x_values is not None:
        x_values = np.asarray(x_values, dtype=float).ravel()
        if x_values.shape[0] != dim:
            raise ValueError(f"x_values must have length {dim}; got {x_values.shape[0]}.")
        x = x_values[sl]
        ticklabels = [f"{val:.3g}" for val in x]
        use_x_values = True
    else:
        x = np.arange(dim_plot)
        use_x_values = False

        if labels is None:
            n_qubits = int(np.log2(dim))
            if 2**n_qubits == dim:
                labels = [format(i, f"0{n_qubits}b") for i in range(dim)]
            else:
                labels = [str(i) for i in range(dim)]

        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if len(labels) != dim:
            raise ValueError(f"labels must have length {dim}; got {len(labels)}.")
        ticklabels = labels[sl]

    if use_x_values and dim_plot > 1:
        spacing = np.diff(x).mean()
        bar_width_actual = spacing * bar_width
    else:
        bar_width_actual = bar_width


    rc = {
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
    }

    col_target = "steelblue"
    col_meas = "darkorange"
    col_cost_pts = "#b94d95"  
    col_best = "steelblue"

    target_alpha = 0.85
    meas_alpha = 0.90
    bar_edge = (0, 0, 0, 0.28)   
    bar_lw = 0.35
    meas_width_factor = 0.55    

    with mpl.rc_context(rc):

        # High-Quality histogram 
        mpl.rcParams["figure.dpi"] = 200         
        mpl.rcParams["path.simplify"] = False     
        mpl.rcParams["patch.antialiased"] = True 

        # ============================================================
        # Figure 1: Distribution Comparison
        # ============================================================
        fig_dist, (ax_before, ax_after) = plt.subplots(1, 2, figsize=figsize_dist)

        # Common tick strategy for continuous x: sparse + rotated
        if use_x_values:
            n_ticks = min(10, dim_plot)
            tick_indices = np.linspace(0, dim_plot - 1, n_ticks, dtype=int)
            xticks = x[tick_indices]
            xticklabels = [ticklabels[i] for i in tick_indices]

        # ---- Before training ----
        ax_before.bar(
            x,
            target[sl],
            width=bar_width_actual,
            alpha=target_alpha,
            label="target",
            zorder=3,
            color=col_target,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_before.bar(
            x,
            before[sl],
            width=bar_width_actual * meas_width_factor,
            alpha=meas_alpha,
            label="measured",
            zorder=2,
            color=col_meas,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_before.set_title(title_before)
        ax_before.set_ylabel(ylabel)
        ax_before.set_xlabel(xlabel)
        ax_before.grid(True, axis="y")
        ax_before.legend(loc="upper right")

        if use_x_values:
            ax_before.set_xticks(xticks)
            ax_before.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=9)
        else:
            ax_before.set_xticks(x)
            ax_before.set_xticklabels(ticklabels, rotation=90, fontsize=8)

        # ---- After training ----
        ax_after.bar(
            x,
            target[sl],
            width=bar_width_actual,
            alpha=target_alpha,
            label="target",
            zorder=3,
            color=col_target,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_after.bar(
            x,
            after[sl],
            width=bar_width_actual * meas_width_factor,
            alpha=meas_alpha,
            label="measured",
            zorder=2,
            color=col_meas,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_after.set_title(title_after)
        ax_after.set_ylabel(ylabel)
        ax_after.set_xlabel(xlabel)
        ax_after.grid(True, axis="y")
        ax_after.legend(loc="upper right")

        if use_x_values:
            ax_after.set_xticks(xticks)
            ax_after.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=9)
        else:
            ax_after.set_xticks(x)
            ax_after.set_xticklabels(ticklabels, rotation=90, fontsize=8)

        if max_states is not None and dim > dim_plot:
            ax_after.annotate(
                f"Showing first {dim_plot} of {dim} states",
                xy=(0.99, 0.02),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=9,
                alpha=0.85,
            )

        fig_dist.tight_layout()

        # ============================================================
        # Figure 2: Cost Evolution
        # ============================================================
        fig_cost, ax_cost = plt.subplots(figsize=figsize_cost)

        steps = np.arange(cost_history.size)

        ax_cost.plot(
            best_idx,
            best_so_far[best_idx],
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            label="best-so-far",
            color=col_best,
            zorder=2,
        )

        ax_cost.plot(
            steps,
            cost_history,
            linestyle="none",
            marker=".",
            markersize=2.2,
            alpha=0.55,
            label="cost",
            color=col_cost_pts,
            zorder=3,
        )

        if cost_log_x:
            ax_cost.set_xscale("log")
        if cost_log_y:
            ax_cost.set_yscale("log")

        ax_cost.set_xlabel(cost_xlabel)
        ax_cost.set_ylabel(cost_ylabel)
        ax_cost.set_title("Training Cost Evolution")
        ax_cost.grid(True, which="both")
        ax_cost.legend(loc="upper right")

        fig_cost.tight_layout()

    return fig_dist, fig_cost