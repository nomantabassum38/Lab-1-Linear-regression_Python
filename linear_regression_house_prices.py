"""
TNK129 Machine Learning for Smart Cities, Lab 1: Linear regression
==================================================================

Model       h_theta(x) = theta_0 + theta_1 * x          (x = living area in m^2)
Loss        J(theta)   = 1/2 * sum_i ( y_i - h_theta(x_i) )^2 = 1/2 * ||X theta - y||^2
Gradient    grad J     = X^T X theta - X^T y
Hessian     H          = X^T X                           (constant, J is quadratic)

Task 1  Normal equations:   X^T X theta* = X^T y
Task 2  Full batch gradient descent from theta = 0, fixed 1e7 iterations
Task 3  Loss surface J(theta) and the path (theta_k, J(theta_k)) of gradient descent
Task 4  Convergence analysis (figures 3 and 4 plus the printed summary)

Usage:  python linear_regression_house_prices.py
        (HousePrices.csv must be in the same folder as this script)

Requires: numpy, pandas, matplotlib
"""

# PART 1: Import libraries. NumPy handles matrix calculations, pandas reads the
# CSV, and Matplotlib draws the results. Path makes file locations independent
# of the folder from which you launch Python; time measures training duration.
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
# Resolve the folder containing THIS Python file, not the current terminal folder.
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "HousePrices.csv"
FIG_DIR = HERE / "figures"

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
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
# PART 2: Load the CSV beside this script and choose the input and target.
# Only SqFt is used as a feature; the other CSV columns are not used in this model.
if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Place HousePrices.csv beside this script: {DATA_FILE}")
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

# ----------------------------------------------------------------------------
# Task 1: normal equations
# ----------------------------------------------------------------------------
# PART 3: Solve directly for the intercept and slope that minimize squared error.
# solve() solves the linear system without explicitly calculating a matrix inverse.
theta_opt = np.linalg.solve(XX, Xy)            # X^T X theta = X^T y
J_opt = loss(theta_opt, X, y)[0]

# ----------------------------------------------------------------------------
# Task 2: gradient descent
# ----------------------------------------------------------------------------
# PART 4: Start at [0, 0] and repeatedly move opposite the loss gradient.
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


def theta_at(k):
    # Used below with recorded iteration counts to retrieve their parameters.
    return thetas[np.searchsorted(record_at, k)]


# Newton's method: theta <- theta - H^{-1} grad J(theta). J is quadratic, so one
# step from any starting point lands exactly on theta*.
theta_newton = np.zeros(2) - np.linalg.solve(XX, XX @ np.zeros(2) - Xy)

# ----------------------------------------------------------------------------
# Convergence theory (used in Task 4)
# ----------------------------------------------------------------------------
# With e_k = theta_k - theta*, GD gives e_{k+1} = (I - alpha X^T X) e_k. Along the
# eigenvector v_j of X^T X the error is multiplied by (1 - alpha lambda_j) per step.
# PART 5: Explain convergence using the curvature of the quadratic loss.
# Large eigenvalues mean steep directions; small ones mean flatter, slower ones.
# A large condition number means these directions converge at very different rates.
lam, V = np.linalg.eigh(XX)                    # lam[0] = lambda_min, lam[1] = lambda_max
kappa = lam[1] / lam[0]
rho = 1.0 - ALPHA * lam                        # contraction factor per iteration
coef0 = V.T @ (np.zeros(2) - theta_opt)        # initial error in the eigenbasis

print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"n = {n} training examples")
print(f"Normal equations : theta* = [{theta_opt[0]:.4f}, {theta_opt[1]:.4f}]   J(theta*) = {J_opt:.6e}")
print(f"GD after {MAX_ITERATIONS:.0e}   : theta  = [{theta_gd[0]:.4f}, {theta_gd[1]:.4f}]   J(theta)  = {J_hist[-1]:.6e}")
print(f"Newton, 1 step   : theta  = [{theta_newton[0]:.4f}, {theta_newton[1]:.4f}]")
print(f"J(theta_GD) - J(theta*) = {gap_hist[-1]:.4e}   (relative: {gap_hist[-1] / J_opt:.2e})")
print()
print(f"Eigenvalues of X^T X : lambda_min = {lam[0]:.4f}, lambda_max = {lam[1]:.4e}")
print(f"Condition number     : kappa = {kappa:.3e}")
print(f"Stability limit      : alpha < 2 / lambda_max = {2 / lam[1]:.3e}")
print(f"Contraction per step : fast direction |1 - alpha lambda_max| = {abs(rho[1]):.4f}")
print(f"                       slow direction  1 - alpha lambda_min  = 1 - {ALPHA * lam[0]:.3e}")
print(f"Slow error left after {MAX_ITERATIONS:.0e} iterations: {rho[0] ** MAX_ITERATIONS:.1%}")
k_needed = np.log(1e-6) / np.log(rho[0])
print(f"Iterations to shrink the slow error by 1e-6: about {k_needed:.2e}")
print("=" * 70)

x_range = np.array([area.min(), area.max()])

# ----------------------------------------------------------------------------
# Figure 1 (Tasks 1 and 2): data, LS fit and the GD hypotheses
# ----------------------------------------------------------------------------
# PART 6: Visualize predictions. Left: houses and fitted lines. Right: the
# difference between GD predictions and the optimal line (zero means agreement).
fig1, (ax, axr) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"wspace": 0.3})

ax.scatter(area, price, s=16, color=C_DATA, alpha=0.8, linewidths=0,
           label=r"training examples $(x^{(i)},\,y^{(i)})$", zorder=2)

first_colors = plt.cm.Blues(np.linspace(0.35, 0.9, 11))
for k in range(11):                            # iteration 0 and the first 10 iterations
    th = theta_at(k)
    ax.plot(x_range, hypothesis(th, x_range), ":", color=first_colors[k], lw=1.5, zorder=3,
            label=r"GD, $h_{\theta_k}(x)$ for $k=0,\dots,10$" if k == 6 else None)
    if k <= 4:                                 # label iterations 0 to 4
        ax.text(x_range.mean(), hypothesis(th, x_range.mean()), rf" $k={k}$ ",
                fontsize=10, color=C_INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=4)

ax.plot(x_range, hypothesis(theta_opt, x_range), color=C_LS, lw=2.2, zorder=5,
        label=r"LS fit $h_{\theta^\star}(x)$ (normal equations)")
ax.plot(x_range, hypothesis(theta_gd, x_range), "--", color=C_GD, lw=1.6, zorder=6,
        label=rf"GD, $h_{{\theta_k}}(x)$ for $k=10^{{{int(np.log10(MAX_ITERATIONS))}}}$")

ax.set_xlabel(r"Living area $x\ [\mathrm{m}^2]$")
ax.set_ylabel(r"Price $y\ [\times 10^{3}]$")
ax.yaxis.set_major_formatter(FuncFormatter(thousands))
ax.set_ylim(-8e3, 2.35e5)
ax.set_title(r"Hypothesis $h_\theta(x)=\theta_0+\theta_1 x$")
ax.legend(loc="upper left", handlelength=2.4)
ax.text(0.98, 0.09,
        rf"$\theta^\star=({theta_opt[0]:.1f},\ {theta_opt[1]:.2f})^{{\!\top}}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12, color=C_LS)

# Right panel: how far each GD hypothesis is from the optimal one
xs = np.linspace(x_range[0], x_range[1], 200)
show = [10, 100, 1_000_000, 2_000_000, 5_000_000, MAX_ITERATIONS]
show_colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(show)))
axr.axhline(0, color=C_LS, lw=2.0, zorder=2)
axr.text(xs[-1], 0, r" $h_{\theta^\star}$", color=C_LS, ha="left", va="bottom", fontsize=13)
for k, col in zip(show, show_colors):
    dev = hypothesis(theta_at(k), xs) - hypothesis(theta_opt, xs)
    axr.plot(xs, dev, color=col, lw=1.8, zorder=3)
    axr.text(xs[-1], dev[-1], " " + iteration_label(k), color=C_INK, fontsize=10.5,
             ha="left", va="center")
axr.set_xlim(xs[0] - 8, xs[-1] + 22)
axr.set_xlabel(r"Living area $x\ [\mathrm{m}^2]$")
axr.set_ylabel(r"$h_{\theta_k}(x)-h_{\theta^\star}(x)$")
axr.set_title(r"Deviation of the GD hypothesis from the LS fit")
fig1.suptitle(rf"Linear regression of house prices: normal equations and gradient descent "
              rf"($\alpha=10^{{{int(np.log10(ALPHA))}}}$)", fontsize=15, y=1.0)
save(fig1, "fig1_hypothesis")

# ----------------------------------------------------------------------------
# Figure 2 (Task 3): loss surface with the GD path (theta_k, J(theta_k))
# ----------------------------------------------------------------------------
# PART 7: Evaluate the loss on a grid of intercept/slope pairs. The 3D surface
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

# ----------------------------------------------------------------------------
# Figure 3 (Tasks 3 and 4): top view of the loss with the GD path
# ----------------------------------------------------------------------------
# PART 8: Show the same landscape from above. Colors represent excess loss;
# the narrow valley explains why GD can improve quickly at first, then slowly.
f0 = np.linspace(range0[0], range0[1], 500)
f1 = np.linspace(range1[0], range1[1], 500)
F0, F1 = np.meshgrid(f0, f1)
gap_grid = suboptimality(np.column_stack((F0.ravel(), F1.ravel())), theta_opt, XX).reshape(F0.shape)
gap_grid = np.maximum(gap_grid, 1e-3)

LABEL_BOX = dict(boxstyle="round,pad=0.3", fc="#1f2933", ec="none", alpha=0.75)
fig3, axc = plt.subplots(figsize=(11, 6.6))
levels = np.geomspace(1e5, gap_grid.max(), 28)
cf = axc.contourf(F0, F1, gap_grid, levels=levels, norm=LogNorm(levels[0], levels[-1]),
                  cmap="viridis", extend="min")
axc.contour(F0, F1, gap_grid, levels=levels[::3], colors="white", linewidths=0.4, alpha=0.5)
cbar = fig3.colorbar(cf, ax=axc, pad=0.02)
cbar.set_label(r"$J(\theta)-J(\theta^\star)$")
cbar.set_ticks(10.0 ** np.arange(5, int(np.log10(gap_grid.max())) + 1))
cbar.ax.set_yticklabels([rf"$10^{{{e}}}$" for e in range(5, int(np.log10(gap_grid.max())) + 1)])

# slow eigen direction through theta*: the floor of the valley
s = np.array([-1.0, 1.0]) * 3e4
axc.plot(theta_opt[0] + s * V[0, 0], theta_opt[1] + s * V[1, 0], "--", color="white", lw=1.0,
         label=r"valley floor: $\theta^\star+t\,v_{\min}$")
axc.plot(thetas[path_mask, 0], thetas[path_mask, 1], color="black", lw=1.5)
axc.plot(thetas[first, 0], thetas[first, 1], "x", color="black", ms=5,
         label=r"GD, first $100$ iterations")
axc.plot(thetas[is_million, 0], thetas[is_million, 1], "o", ms=6.5, mfc="white", mec="black",
         mew=1.2, label=r"GD, every $10^{6}$ iterations")
axc.plot(0, 0, "o", ms=9, color="black", label=r"start $\theta_0=\mathbf{0}$")
axc.plot(theta_opt[0], theta_opt[1], marker="*", ms=22, ls="none", mfc=C_LS, mec="white",
         mew=0.9, label=r"optimum $\theta^\star$")

for k in (1_000_000, 5_000_000, MAX_ITERATIONS):
    th = theta_at(k)
    axc.annotate(iteration_label(k), xy=th, xytext=(0, 26), textcoords="offset points",
                 ha="center", fontsize=11, color="white", bbox=LABEL_BOX,
                 arrowprops=dict(arrowstyle="-", color="black", lw=0.8, shrinkB=5))
axc.annotate(r"fast phase: $k\leq 10$", xy=(0, 0.5 * theta_at(10)[1]), xytext=(42, 0),
             textcoords="offset points", va="center", fontsize=11, color="white", bbox=LABEL_BOX,
             arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

axc.set_xlim(range0)
axc.set_ylim(range1)
axc.set_xlabel(r"$\theta_0$")
axc.set_ylabel(r"$\theta_1$")
axc.grid(False)
axc.set_title(r"Level sets of $J(\theta)$ and the gradient descent path "
              rf"($\kappa(X^{{\!\top}}X)\approx{kappa / 10 ** int(np.log10(kappa)):.1f}"
              rf"\times10^{{{int(np.log10(kappa))}}}$)")
axc.legend(loc="upper right", labelcolor="white", frameon=True, facecolor="#1f2933",
           edgecolor="none", framealpha=0.75, borderpad=0.8)
save(fig3, "fig3_loss_contours")

# ----------------------------------------------------------------------------
# Figure 4 (Task 4): convergence of J(theta_k) and of theta_k
# ----------------------------------------------------------------------------
# PART 9: Compare observed convergence with the quadratic-loss prediction,
# then plot the relative error in each parameter. Log axes show many time scales.
kk = record_at[1:].astype(float)               # skip k = 0 on logarithmic axes
gap_theory = 0.5 * ((coef0 ** 2 * lam)[None, :] * rho[None, :] ** (2 * kk[:, None])).sum(axis=1)

fig4, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"wspace": 0.28})

axa.loglog(kk, gap_hist[1:], color=C_GD, lw=2.2, label=r"gradient descent")
axa.loglog(kk, gap_theory, "--", color=C_LS, lw=1.3,
           label=r"theory: $\frac{1}{2}\sum_{j}\lambda_j\,c_j^{2}\,(1-\alpha\lambda_j)^{2k}$")
axa.set_xlabel(r"Iteration $k$")
axa.set_ylabel(r"$J(\theta_k)-J(\theta^\star)$")
axa.set_title(r"Suboptimality of the loss")
axa.legend(loc="upper right")
axa.annotate(r"fast direction $v_{\max}$:" "\n" rf"factor $|1-\alpha\lambda_{{\max}}|\approx{abs(rho[1]):.2f}$ per step",
             xy=(4, gap_hist[np.searchsorted(record_at, 4)]), xytext=(20, 6e9), fontsize=11,
             arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.8), color=C_INK)
axa.annotate(r"slow direction $v_{\min}$:" "\n" rf"factor $1-\alpha\lambda_{{\min}}\approx 1-{ALPHA * lam[0] / 1e-7:.1f}\times10^{{-7}}$",
             xy=(3e4, gap_hist[np.searchsorted(record_at, 30000)]), xytext=(60, 5e8), fontsize=11,
             arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.8), color=C_INK)

err0 = np.abs(thetas[1:, 0] - theta_opt[0]) / abs(theta_opt[0])
err1 = np.abs(thetas[1:, 1] - theta_opt[1]) / abs(theta_opt[1])
axb.loglog(kk, err0, color=C_GD, lw=2.2,
           label=r"intercept: $|\theta_{0,k}-\theta_0^\star|\,/\,|\theta_0^\star|$")
axb.loglog(kk, err1, color="#2a9d8f", lw=2.2,
           label=r"slope: $|\theta_{1,k}-\theta_1^\star|\,/\,|\theta_1^\star|$")
axb.set_xlabel(r"Iteration $k$")
axb.set_ylabel(r"Relative parameter error")
axb.set_title(r"Convergence of the parameters $\theta_k$")
axb.legend(loc="lower left")
axb.text(kk[-1], err0[-1], rf"  ${err0[-1]:.0%}$".replace("%", r"\%"), va="center", fontsize=11)
axb.text(kk[-1], err1[-1], rf"  ${err1[-1]:.1%}$".replace("%", r"\%"), va="center", fontsize=11)
axb.set_xlim(right=kk[-1] * 6)
fig4.suptitle(rf"Convergence of gradient descent ($\alpha=10^{{{int(np.log10(ALPHA))}}}$, "
              rf"$\lambda_{{\min}}={lam[0]:.2f}$, $\lambda_{{\max}}={lam[1] / 1e6:.2f}\times10^{{6}}$)",
              fontsize=15, y=1.0)
save(fig4, "fig4_convergence")

# PART 10: Each figure has been saved as PNG and PDF in the figures subfolder.
# show() opens the interactive plot windows when using a graphical backend.
print(f"\nFigures saved in: {FIG_DIR}")
plt.show()
