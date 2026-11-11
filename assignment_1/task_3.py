"""
Task 3 — Loss surface and gradient descent path
Evaluate $J(\theta)$ on a grid of intercept/slope pairs and draw the 3D loss surface. Overlay the initial point, the first 100 steps, every millionth step, and the optimum from the normal equations. The path shows how the parameters move down the surface.

Run: python3 task_3.py
This file loads its own data and computes all prerequisites.
"""

import numpy as np
from pathlib import Path


def gradient_descent(XX, Xy, alpha, n_iter, record_at):
    """Full batch gradient descent for the least squares loss, started at theta = 0."""
    a, b, d = float(XX[0, 0]), float(XX[0, 1]), float(XX[1, 1])
    c0, c1 = float(Xy[0]), float(Xy[1])

    record_at = np.asarray(record_at, dtype=np.int64)
    history = np.empty((record_at.size, 2))
    t0 = t1 = 0.0
    ptr = 0
    if record_at[0] == 0:
        history[0] = (t0, t1)
        ptr = 1
    next_k = int(record_at[ptr]) if ptr < record_at.size else -1

    for k in range(1, n_iter + 1):
        g0 = a * t0 + b * t1 - c0
        g1 = b * t0 + d * t1 - c1
        t0 -= alpha * g0
        t1 -= alpha * g1
        if k == next_k:
            history[ptr] = (t0, t1)
            ptr += 1
            next_k = int(record_at[ptr]) if ptr < record_at.size else -1
    return history


def loss(theta, X, y):
    """J(theta) = 1/2 ||X theta - y||^2 for one theta (2,) or many thetas (m, 2)."""
    theta = np.atleast_2d(theta)
    predictions = theta[:, 0, None] * X[None, :, 0] + theta[:, 1, None] * X[None, :, 1]
    r = predictions - y
    return 0.5 * np.sum(r * r, axis=1)


def suboptimality(theta, theta_opt, XX):
    """Exact quadratic gap between a parameter vector and the optimum."""
    e = np.atleast_2d(theta) - theta_opt
    return 0.5 * np.einsum("ij,jk,ik->i", e, XX, e)


def hypothesis(theta, x):
    """h_theta(x) = theta_0 + theta_1 x."""
    return theta[0] + theta[1] * np.asarray(x)


def iteration_label(k):
    """Compact LaTeX label for an iteration count."""
    k = int(k)
    if k < 1000:
        return rf"$k={k}$"
    exponent = int(np.floor(np.log10(k)))
    mantissa = k / 10 ** exponent
    if np.isclose(mantissa, 1.0):
        return rf"$k=10^{{{exponent}}}$"
    return rf"$k={mantissa:g}\times10^{{{exponent}}}$"


def thousands(v, _pos=None):
    return f"{v / 1e3:g}"


def save(fig, name, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png")
    fig.savefig(fig_dir / f"{name}.pdf")


def main():
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # ----------------------------------------------------------------------------
    # Settings
    # ----------------------------------------------------------------------------
    # Load data beside this task file, regardless of the terminal working directory.
    HERE = Path(__file__).resolve().parent
    DATA_FILE = HERE / "HousePrices.csv"
    FIG_DIR = HERE / "figures" / "task_3"

    ALPHA = 1e-7                  # learning rate (try other values later)
    MAX_ITERATIONS = 10_000_000   # fixed number of GD iterations (no stopping criterion)
    SQFT_PER_M2 = 3.28084 ** 2    # 1 m = 3.28084 ft

    # Colours
    C_DATA = "#7b8794"    # training examples
    C_LS = "#d1432f"      # least squares (normal equations)
    C_GD = "#12355b"      # gradient descent
    C_INK = "#1f2933"     # text and axes

    # Apply a consistent appearance to all figures; these options do not affect fitting.
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "axes.edgecolor": C_INK,
        "axes.labelcolor": C_INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d9dee4",
        "grid.linewidth": 0.6,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


    # ----------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------
    def gradient_descent(XX, Xy, alpha, n_iter, record_at):
        """Full batch gradient descent for the least squares loss, started at theta = 0.

        Update rule:  theta <- theta - alpha * (X^T X theta - X^T y)

        X^T X and X^T y are computed once, so one iteration costs O(d^2) instead of
        O(n d). For this lab d = 1 (two parameters), and the 2 x 2 matrix vector
        product is written out with scalars. This is mathematically identical to
        `theta = theta - alpha * (XX @ theta - Xy)` but far faster in pure Python
        for 1e7 iterations.

        Returns the iterates theta_k for every k listed in `record_at`.
        """
        a, b, d = float(XX[0, 0]), float(XX[0, 1]), float(XX[1, 1])
        c0, c1 = float(Xy[0]), float(Xy[1])

        # Store only selected iterations to avoid keeping ten million parameter pairs.
        record_at = np.asarray(record_at, dtype=np.int64)
        history = np.empty((record_at.size, 2))
        t0 = t1 = 0.0                               # theta = 0
        ptr = 0
        if record_at[0] == 0:
            history[0] = (t0, t1)
            ptr = 1
        next_k = int(record_at[ptr]) if ptr < record_at.size else -1

        for k in range(1, n_iter + 1):
            # Compute both gradient components using the same current parameter pair.
            g0 = a * t0 + b * t1 - c0               # gradient, component theta_0
            g1 = b * t0 + d * t1 - c1               # gradient, component theta_1
            t0 -= alpha * g0                        # theta <- theta - alpha * grad
            t1 -= alpha * g1
            if k == next_k:
                history[ptr] = (t0, t1)
                ptr += 1
                next_k = int(record_at[ptr]) if ptr < record_at.size else -1
        return history


    def loss(theta, X, y):
        """J(theta) = 1/2 ||X theta - y||^2 for one theta (2,) or many thetas (m, 2)."""
        theta = np.atleast_2d(theta)
        # Each row contains predicted prices minus observed prices for one parameter pair.
        # For this two-parameter model, calculate intercept + slope * area directly.
        # Broadcasting evaluates every parameter pair against every training example.
        predictions = theta[:, 0, None] * X[None, :, 0] + theta[:, 1, None] * X[None, :, 1]
        r = predictions - y
        return 0.5 * np.sum(r * r, axis=1)


    def suboptimality(theta, theta_opt, XX):
        """J(theta) - J(theta*) = 1/2 (theta - theta*)^T X^T X (theta - theta*).

        Exact for a quadratic loss and numerically stable close to the optimum
        (it avoids subtracting two numbers of size 1e10).
        """
        e = np.atleast_2d(theta) - theta_opt
        return 0.5 * np.einsum("ij,jk,ik->i", e, XX, e)


    def hypothesis(theta, x):
        """h_theta(x) = theta_0 + theta_1 x."""
        return theta[0] + theta[1] * np.asarray(x)


    def iteration_label(k):
        """Compact LaTeX label for an iteration count."""
        k = int(k)
        if k < 1000:
            return rf"$k={k}$"
        exponent = int(np.floor(np.log10(k)))
        mantissa = k / 10 ** exponent
        if np.isclose(mantissa, 1.0):
            return rf"$k=10^{{{exponent}}}$"
        return rf"$k={mantissa:g}\times10^{{{exponent}}}$"


    def thousands(v, _pos=None):
        return f"{v / 1e3:g}"


    def save(fig, name):
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png")
        fig.savefig(FIG_DIR / f"{name}.pdf")


    # ----------------------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------------------
    # Load the CSV beside this task file and choose the input and target.
    # Only SqFt is used as a feature; the other CSV columns are not used in this model.
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Place HousePrices.csv beside this task file: {DATA_FILE}")
    print(f"Loading house prices from: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    print(data.head(5).to_string(index=False), "\n")

    area = data["SqFt"].to_numpy(dtype=float) / SQFT_PER_M2    # living area [m^2]
    price = data["Price"].to_numpy(dtype=float)                # price [original currency]

    n = price.size                 # number of training examples
    d = 1                          # number of input features (without the intercept)

    theta = np.zeros(d + 1)                        # theta[0] is the bias theta_0
    # Each row of X is [1, area]. The 1 allows the model to learn an intercept:
    # predicted price = theta[0] + theta[1] * area. X has shape (n, 2).
    X = np.column_stack((np.ones(n), area))        # design matrix, X theta ~ y
    y = price                                      # targets

    # Precompute the matrices reused by both fitting methods. The loss is half the
    # SUM of squared errors (not the mean), so its gradient has no division by n.
    XX = X.T @ X                   # Hessian of J
    Xy = X.T @ y

    # Solve directly for the intercept and slope that minimize squared error.
    # solve() solves the linear system without explicitly calculating a matrix inverse.
    theta_opt = np.linalg.solve(XX, Xy)            # X^T X theta = X^T y
    J_opt = loss(theta_opt, X, y)[0]

    x_range = np.array([area.min(), area.max()])

    # Start at [0, 0] and repeatedly move opposite the loss gradient.
    # ALPHA controls the step size; this run always performs MAX_ITERATIONS updates.
    # Record early steps, logarithmically spaced steps, and each millionth step.
    millions = np.arange(1, MAX_ITERATIONS // 1_000_000 + 1) * 1_000_000
    record_at = np.unique(np.concatenate((
        np.arange(0, 101),                                           # first 100 iterations
        np.round(np.geomspace(100, MAX_ITERATIONS, 300)).astype(np.int64),   # for log plots
        millions,                                                    # every million
        [MAX_ITERATIONS],
    )))

    print(f"Running gradient descent: alpha = {ALPHA:g}, {MAX_ITERATIONS:,} iterations ...")
    tic = time.perf_counter()
    thetas = gradient_descent(XX, Xy, ALPHA, MAX_ITERATIONS, record_at)
    print(f"done in {time.perf_counter() - tic:.1f} s\n")

    # Compare the final GD fit with the direct optimum. J_hist stores the loss;
    # gap_hist stores the extra loss above the best achievable value.
    theta_gd = thetas[-1]
    J_hist = loss(thetas, X, y)
    gap_hist = suboptimality(thetas, theta_opt, XX)

    from training_animation import save_training_gif
    save_training_gif(area, price, theta_opt, record_at, thetas,
                      FIG_DIR / "task3_training.gif",
                      "Task 3: Gradient descent training")



    def theta_at(k):
        # Used below with recorded iteration counts to retrieve their parameters.
        return thetas[np.searchsorted(record_at, k)]
    print(f"GD parameters: {theta_gd}")
    print(f"GD loss: {J_hist[-1]:.6e}")
    print(f"Excess loss above optimum: {gap_hist[-1]:.6e}")

    # Evaluate the loss on a grid of intercept/slope pairs. The 3D surface
    # shows the optimization landscape, and the markers trace the recorded GD steps.
    range0 = np.sort([-0.5 * theta_opt[0], 1.5 * theta_opt[0]])     # same ranges as the Matlab starter
    range1 = np.sort([-0.1 * theta_opt[1], 2.0 * theta_opt[1]])

    g0 = np.linspace(range0[0], range0[1], 60)
    g1 = np.linspace(range1[0], range1[1], 60)
    T0, T1 = np.meshgrid(g0, g1)
    J_grid = loss(np.column_stack((T0.ravel(), T1.ravel())), X, y).reshape(T0.shape)

    first = record_at <= 100
    is_million = np.isin(record_at, millions)
    path_mask = first | is_million                 # the points the lab asks us to plot

    fig2 = plt.figure(figsize=(11, 8))
    ax3 = fig2.add_subplot(projection="3d", computed_zorder=False)
    ax3.plot_surface(T0, T1, J_grid, cmap="viridis", alpha=0.72, rstride=2, cstride=2,
                     edgecolor=(1, 1, 1, 0.35), linewidth=0.3, antialiased=True, zorder=1)
    ax3.contour(T0, T1, J_grid, levels=np.geomspace(J_grid.min() * 1.05, J_grid.max(), 14),
                zdir="z", offset=0, cmap="viridis", linewidths=0.8, zorder=0)

    ax3.plot(thetas[path_mask, 0], thetas[path_mask, 1], J_hist[path_mask],
             color="black", lw=1.4, zorder=5)
    ax3.plot(thetas[first, 0], thetas[first, 1], J_hist[first], "x", color="black", ms=5,
             zorder=6, label=r"GD, first $100$ iterations")
    ax3.plot(thetas[is_million, 0], thetas[is_million, 1], J_hist[is_million], "o", ms=6,
             mfc="white", mec="black", mew=1.2, zorder=7,
             label=r"GD, every $10^{6}$ iterations")
    ax3.plot([0], [0], [J_hist[0]], "o", ms=9, color="black", zorder=8,
             label=r"start $\theta_0=\mathbf{0}$")
    ax3.plot([theta_opt[0]], [theta_opt[1]], [J_opt], marker="*", ms=20, ls="none", mfc=C_LS,
             mec="white", mew=0.8, zorder=9, label=r"optimum $\theta^\star$ (normal equations)")

    ax3.set_xlabel(r"$\theta_0$", labelpad=12)
    ax3.set_ylabel(r"$\theta_1$", labelpad=10)
    ax3.set_zlabel(r"$J(\theta)\ [\times 10^{12}]$", labelpad=10)
    ax3.zaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / 1e12:.1f}"))
    ax3.xaxis.set_major_locator(MaxNLocator(5))
    ax3.yaxis.set_major_locator(MaxNLocator(6))
    ax3.set_zlim(0, J_grid.max())
    ax3.view_init(elev=30, azim=-62)
    ax3.set_title(r"Loss surface $J(\theta)=\frac{1}{2}\sum_{i=1}^{n}\left(y^{(i)}-h_\theta(x^{(i)})\right)^{2}$"
                  r" and the path $(\theta_k,\,J(\theta_k))$ of gradient descent", pad=4)
    ax3.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
    save(fig2, "fig2_loss_surface")
    plt.show()


if __name__ == "__main__":
    main()
