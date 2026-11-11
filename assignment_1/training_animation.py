"""Save labelled training frames without requiring a GUI or external GIF encoder."""

from pathlib import Path
import os
import shutil
import tempfile

import numpy as np


def publish_gif(local_file, output):
    """Publish a completed GIF atomically; retain the local file on storage errors."""
    staged = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".gif", delete=False) as f:
            staged = Path(f.name)
        shutil.copyfile(local_file, staged)
        os.replace(staged, output)
    except OSError as error:
        print(f"Could not copy GIF to {output}: {error}")
        print(f"Training GIF is available locally at: {local_file}")
        return local_file
    finally:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
    shutil.rmtree(local_file.parent)
    print(f"Training GIF saved to: {output}")
    return output


def save_training_gif(area, price, theta_opt, iterations, thetas, output, title,
                      direct_solve=False):
    """Show every early update and sampled later updates from the actual history.

    Ten million GIF frames would be impractical. Preserve iterations 0–100,
    then select up to 60 of the remaining recorded steps, including the last.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    iterations = np.asarray(iterations)
    thetas = np.asarray(thetas)
    early = np.flatnonzero(iterations <= 100)
    later = np.flatnonzero(iterations > 100)
    if later.size:
        later = later[np.unique(np.linspace(0, later.size - 1, min(60, later.size)).astype(int))]
    indices = np.concatenate((early, later))
    losses = 0.5 * np.sum(
        (thetas[:, 0, None] + thetas[:, 1, None] * area - price) ** 2, axis=1)
    x = np.array([area.min(), area.max()])
    fig = plt.figure(figsize=(12, 5.8))
    fit_ax = fig.add_subplot(1, 2, 1)
    surface_ax = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
    fig.subplots_adjust(bottom=0.17, top=0.77, left=0.08, right=0.92, wspace=0.22)
    fig.suptitle(title)
    fit_ax.scatter(area, price, s=14, color="#7b8794", label="Observed prices")
    fit_ax.plot(x, theta_opt[0] + theta_opt[1] * x, "--", color="#d1432f", label="Optimal fit")
    fitted, = fit_ax.plot([], [], color="#12355b", lw=2, label="Current model")
    fit_ax.set(xlabel="Living area [m²]", ylabel="Price",
               ylim=(min(0, price.min()) - 0.05 * np.ptp(price), price.max() * 1.1))
    fit_ax.legend(fontsize=9)

    # Include the entire recorded trajectory and optimum within fixed limits.
    parameters = np.vstack((thetas, theta_opt))
    lower, upper = parameters.min(axis=0), parameters.max(axis=0)
    padding = np.maximum((upper - lower) * 0.15, 1.0)
    t0, t1 = np.meshgrid(np.linspace(lower[0] - padding[0], upper[0] + padding[0], 45),
                         np.linspace(lower[1] - padding[1], upper[1] + padding[1], 45))
    residual = t0[..., None] + t1[..., None] * area - price
    surface_loss = 0.5 * np.sum(residual ** 2, axis=-1)
    # Log loss makes the narrow valley and late improvements visible together.
    z = np.log10(np.maximum(surface_loss, np.finfo(float).tiny))
    path_z = np.log10(np.maximum(losses, np.finfo(float).tiny))
    optimal_loss = 0.5 * np.sum((theta_opt[0] + theta_opt[1] * area - price) ** 2)
    optimal_z = np.log10(max(optimal_loss, np.finfo(float).tiny))
    surface_ax.plot_surface(t0, t1, z, cmap="viridis", alpha=0.5,
                            linewidth=0, antialiased=True, zorder=1)
    surface_ax.plot([theta_opt[0]], [theta_opt[1]], [optimal_z], "*",
                    color="#d1432f", markersize=13, label="Optimum", zorder=5)
    trail, = surface_ax.plot([], [], [], color="black", lw=2, zorder=4,
                             label="Direct solve" if direct_solve else "Training path")
    dot, = surface_ax.plot([], [], [], "o", color="#ff8c00", markersize=7, zorder=6)
    surface_ax.set(xlabel=r"Intercept $\theta_0$", ylabel=r"Slope $\theta_1$", zlabel=r"$\log_{10}$(loss)",
                   xlim=(t0.min(), t0.max()), ylim=(t1.min(), t1.max()),
                   zlim=(min(optimal_z, z.min()) - 0.08, z.max() + 0.08))
    surface_ax.set_title("3D loss surface", fontsize=12)
    surface_ax.tick_params(labelsize=8)
    surface_ax.legend(loc="upper left", fontsize=8)
    surface_ax.set_box_aspect((1.2, 1, 0.8))
    if direct_solve:
        # Rotate around the two real states; do not invent intermediate updates.
        indices = np.repeat([0, len(thetas) - 1], [20, 40])
    status = fig.text(0.5, 0.85, "", ha="center", fontsize=10)
    if not direct_solve:
        fig.text(0.5, 0.02, "Every iteration through 100; sampled iterations afterwards (see counter).",
                 ha="center", fontsize=9)

    def update(frame):
        i = indices[frame]
        theta = thetas[i]
        fitted.set_data(x, theta[0] + theta[1] * x)
        trail.set_data_3d(thetas[:i + 1, 0], thetas[:i + 1, 1], path_z[:i + 1])
        dot.set_data_3d([theta[0]], [theta[1]], [path_z[i]])
        surface_ax.view_init(elev=28, azim=-65 + 110 * frame / max(1, len(indices) - 1))
        step = ("Initial model" if i == 0 else "Normal equations solved") if direct_solve else f"Iteration {iterations[i]:,} / {iterations[-1]:,}"
        status.set_text(f"{step}    |    loss = {losses[i]:.4e}\n"
                        f"intercept = {theta[0]:.2f}    slope = {theta[1]:.2f}")
        return fitted, trail, dot, status

    output = Path(output)
    animation = FuncAnimation(fig, update, frames=len(indices), interval=100,
                              repeat=True, blit=False)
    # Encode outside cloud storage: Pillow performs many writes while saving.
    # Keep this directory if publishing fails so the completed GIF remains usable.
    local_file = Path(tempfile.mkdtemp(prefix="ml-training-")) / output.name
    try:
        animation.save(local_file, writer=PillowWriter(fps=10), dpi=85)
    except Exception:
        shutil.rmtree(local_file.parent, ignore_errors=True)
        raise
    finally:
        plt.close(fig)
    saved = publish_gif(local_file, output)
    # Also provide a still 3D overview for reports alongside the animation.
    # Save beside the local fallback if the cloud destination was unavailable.
    update(len(indices) - 1)
    still = saved.with_name(output.stem + "_3d.png")
    try:
        fig.savefig(still, dpi=150, bbox_inches="tight")
    except OSError as error:
        print(f"Could not save 3D overview: {error}")
    return saved
