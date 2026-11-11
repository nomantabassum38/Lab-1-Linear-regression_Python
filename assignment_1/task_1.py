"""
Task 1 — Normal equations
Solve $X^T X\theta^*=X^T y$ directly for the least-squares optimum. `np.linalg.solve` avoids explicitly forming a matrix inverse. The scatter plot shows observed prices and the optimal fitted line.

Run: python3 task_1.py (save plots), or python3 task_1.py --show (open a window).
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
    import argparse
    import os
    import tempfile
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Fit house prices and save the plot.")
    parser.add_argument("--show", action="store_true", help="also open an interactive plot window")
    args = parser.parse_args()

    # Use a writable cache before importing Matplotlib. Saving plots needs no GUI.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ml-assignments-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ml-assignments-cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")

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
    FIG_DIR = HERE / "figures" / "task_1"

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
            t0 -= alpha * g0                        # theta <- theta - alpha * gradj
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
    print(f"Optimal intercept: {theta_opt[0]:.4f}")
    print(f"Optimal slope: {theta_opt[1]:.4f}")
    print(f"Minimum loss: {J_opt:.6e}")
    x_range = np.array([area.min(), area.max()])

    from training_animation import save_training_gif
    save_training_gif(area, price, theta_opt, [0, 1],
                      np.array([np.zeros(2), theta_opt]),
                      FIG_DIR / "task1_training.gif", "Task 1: Normal equations",
                      direct_solve=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(area, price, color=C_DATA, label="Observed prices")
    ax.plot(x_range, hypothesis(theta_opt, x_range), color=C_LS, label="Normal equations")
    ax.set(xlabel="Living area [m²]", ylabel="Price", title="Task 1: Least-squares fit")
    ax.legend()
    save(fig, "task1_normal_equations")
    print(f"Plots saved to: {FIG_DIR}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
