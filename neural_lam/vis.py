# Third-party
import matplotlib
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

# First-party
from neural_lam import constants, utils


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, title=None, step_length=3):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = (
            f"{error:.3f}" if 0.001 < error < 9999 else f"{error:.2E}"
        )
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in steps
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Step", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        f"{name} ({unit})"
        for name, unit in zip(
            constants.PARAM_NAMES_SHORT, constants.PARAM_UNITS
        )
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(pred, target, obs_mask, title=None, vrange=None):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Set up masking of earth
    mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
    pixel_alpha = mask_reshaped.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    for ax in axes:
        ax.set_facecolor("white")

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        data_grid = data.reshape(*constants.GRID_SHAPE).cpu().numpy()
        im = ax.imshow(
            data_grid,
            origin="lower",
            extent=constants.GRID_LIMITS,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

        # Plot inner magnetosphere boundary
        earth_circle = mp.Circle(
            (0, 0), 4.55, color="black", fill=False, linewidth=1.5
        )
        ax.add_patch(earth_circle)

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    axes[0].set_xlabel(r"$x\ (R_E)$", size=10)
    axes[0].set_ylabel(r"$z\ (R_E)$", size=10)
    axes[1].set_xlabel(r"$x\ (R_E)$", size=10)
    axes[1].set_ylabel(r"$z\ (R_E)$", size=10)
    cbar = fig.colorbar(im, aspect=20)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(error, obs_mask, title=None, vrange=None):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    # Set up masking of earth
    mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
    pixel_alpha = mask_reshaped.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.set_facecolor("white")

    error_grid = error.reshape(*constants.GRID_SHAPE).cpu().numpy()

    im = ax.imshow(
        error_grid,
        origin="lower",
        extent=constants.GRID_LIMITS,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    earth_circle = mp.Circle(
        (0, 0), 4.55, color="black", fill=False, linewidth=1.25
    )
    ax.add_patch(earth_circle)

    # Ticks and labels
    ax.set_xlabel(r"$x\ (R_E)$", size=10)
    ax.set_ylabel(r"$z\ (R_E)$", size=10)
    cbar = fig.colorbar(im, aspect=20)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
