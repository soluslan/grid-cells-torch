"""Every plot the evaluation produces.

Split out from scores.py (which measures, and imports no matplotlib) and
evaluate.py (which runs the model and wires things together), so that adding
a panel never touches the code that produces the numbers.

Deliberately does NOT call matplotlib.use(): a notebook wants its own
inline backend. The headless CLI selects "Agg" itself before importing this.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from scores import circle_mask

# cmap="jet" matches the paper's Fig. 1d colour scheme.
CMAP = "jet"


# --------------------------------------------------------------------------
# Single panels -- one unit, one row of Fig. 1d
# --------------------------------------------------------------------------

def sac_plotting_mask(nbins):
    """NaN outside the region of the autocorrelogram that is ever populated."""
    return np.where(circle_mask([nbins * 2 - 1] * 2, nbins), 1.0, np.nan)


def plot_ratemap(ratemap, ax=None, title=None, **kwargs):
    ax = ax or plt.gca()
    ax.imshow(ratemap, interpolation="none", cmap=kwargs.pop("cmap", CMAP), **kwargs)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=8)
    return ax


def plot_sac(sac, nbins, mask_params=None, ax=None, title=None, **kwargs):
    """Autocorrelogram, with the winning annulus drawn on if given."""
    ax = ax or plt.gca()
    ax.imshow(sac * sac_plotting_mask(nbins), interpolation="none",
              cmap=kwargs.pop("cmap", CMAP), **kwargs)
    if mask_params is not None:
        # mask_params are in bins, the same units as the SAC image axes.
        centre = nbins - 1
        for radius in mask_params:
            ax.add_artist(plt.Circle((centre, centre), radius, fill=False,
                                     edgecolor="k", lw=0.6))
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=8)
    return ax


def plot_hd_tuning(tuning, ax=None, title=None, color="#1a4f7a"):
    """Polar plot of mean activity vs head direction -- Fig. 1d bottom row.

    `tuning` is mean activity per angular bin (the paper uses 20). Shifted by
    its own minimum before drawing, for the same reason the resultant-vector
    measure shifts: a linear layer's activations are signed, and a negative
    radius is not something a polar axis can show.
    """
    ax = ax or plt.gca()
    r = np.asarray(tuning, dtype=float)
    r = r - r.min()
    n = len(r)
    theta = -np.pi + (np.arange(n) + 0.5) * (2 * np.pi / n)  # bin centres
    ax.plot(np.append(theta, theta[0]), np.append(r, r[0]), color=color, lw=1.2)
    ax.fill(np.append(theta, theta[0]), np.append(r, r[0]), color=color, alpha=0.25)
    # All tick positions must be positive: a negative one widens the theta
    # limits past a full turn (-90deg..360deg) and the plot renders as a wedge.
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(alpha=0.3, lw=0.4)
    if title is not None:
        ax.set_title(title, fontsize=8)
    return ax


# --------------------------------------------------------------------------
# Grids of units
# --------------------------------------------------------------------------

def plot_unit_grid(ratemaps, sacs, mask_params, scores, nbins, indices=None,
                   cols=8, threshold=None, hd_tuning=None, panel_size=1.6,
                   title=None):
    """Units as columns of vertically-adjacent panels, in the layout of Fig. 1d.

    ratemap on top, autocorrelogram below it, polar plot below that when
    `hd_tuning` is supplied. Units above `threshold` are boxed in red.

    `indices` selects and orders which units to draw -- pass the top N by
    score for a preview, or every unit for the full export.
    """
    if indices is None:
        indices = np.argsort(-np.asarray(scores))
    indices = list(indices)
    has_polar = hd_tuning is not None
    rows_per_unit = 3 if has_polar else 2

    blocks = int(np.ceil(len(indices) / cols))
    total_rows = blocks * rows_per_unit
    fig = plt.figure(figsize=(cols * panel_size, total_rows * panel_size))
    if title:
        fig.suptitle(title, fontsize=12, y=1.0)

    for pos, unit in enumerate(indices):
        block, col = divmod(pos, cols)
        base = block * rows_per_unit

        def cell(row, polar=False):
            return fig.add_subplot(total_rows, cols, (base + row) * cols + col + 1,
                                   projection="polar" if polar else None)

        rf = cell(0)
        label = f"{unit} ({scores[unit]:.2f})"
        plot_ratemap(ratemaps[unit], ax=rf, title=label)
        acr = cell(1)
        plot_sac(sacs[unit], nbins, mask_params=mask_params[unit], ax=acr)
        axes = [rf, acr]
        if has_polar:
            axes.append(plot_hd_tuning(hd_tuning[unit], ax=cell(2, polar=True)))

        if threshold is not None and scores[unit] > threshold:
            rf.title.set_color("red")
            rf.title.set_fontweight("bold")
            for ax in (rf, acr):
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       fill=False, edgecolor="red", linewidth=2))

    fig.tight_layout()
    return fig


def save_unit_pdf(out_path, **kwargs):
    """plot_unit_grid straight to a PDF, without leaving the figure open."""
    fig = plot_unit_grid(**kwargs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_score_distribution(scores_by_layer, threshold, ax=None, bins=60):
    """Where each layer's units sit relative to the grid-like cutoff."""
    ax = ax or plt.gca()
    colors = {"bottleneck": "#1a4f7a", "lstm": "#c0392b"}
    for name, scores in scores_by_layer.items():
        n = int((np.asarray(scores) > threshold).sum())
        ax.hist(scores, bins=bins, alpha=0.6, color=colors.get(name),
                label=f"{name}: {n}/{len(scores)} > {threshold}")
    ax.axvline(threshold, color="k", ls="--", lw=1)
    ax.set_xlabel("gridness"); ax.set_ylabel("units")
    ax.legend(fontsize=9)
    return ax


# --------------------------------------------------------------------------
# Cell types -- Fig. 1e,g and Ext. Data Fig. 3b
# --------------------------------------------------------------------------

def plot_scale_distribution(scales, means=None, bics=None, ax=None, bins=20):
    """Fig. 1e: where grid scales sit, and the mixture BIC chose.

    Cluster centres are annotated with the ratio between neighbours -- the
    paper's actual claim is about that ratio (~1.5), not the absolute scales,
    since those depend on the arena.
    """
    ax = ax or plt.gca()
    s = np.asarray(scales, dtype=float)
    s = s[np.isfinite(s)]
    ax.hist(100 * s, bins=bins, color="#1a4f7a", alpha=0.65, density=True)

    if means is not None and len(means):
        # Capture ylim once: reading it back per-annotation lets the first
        # annotations move the axis under the later ones.
        top = ax.get_ylim()[1]
        for m in means:
            ax.axvline(100 * m, color="#c0392b", ls="--", lw=1.2)
            ax.text(100 * m, top * 0.99, f"{100*m:.0f}", ha="center", va="top",
                    fontsize=8, color="#c0392b")
        # Ratio arrows sit at 45% height, clear of the BIC inset above them --
        # at 80% the inset hid every arrow but the first.
        for a, b in zip(means[:-1], means[1:]):
            ax.annotate("", xy=(100 * b, top * 0.45), xytext=(100 * a, top * 0.45),
                        arrowprops=dict(arrowstyle="<->", lw=0.9, color="#333333"),
                        zorder=5)
            ax.text(50 * (a + b), top * 0.47, f"x{b/a:.2f}", ha="center",
                    fontsize=8, color="#333333", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xlabel("grid scale (cm)")
    ax.set_ylabel("probability density")
    ax.set_title(f"n={len(s)} grid-like units"
                 + (f", BIC -> {len(means)} clusters" if means is not None and len(means) else ""),
                 fontsize=10)

    if bics is not None and len(bics):
        inset = ax.inset_axes([0.70, 0.62, 0.27, 0.32])
        ks = np.arange(1, len(bics) + 1)
        inset.plot(ks, bics, "o-", ms=3, lw=1, color="#1a4f7a")
        inset.plot(ks[np.argmin(bics)], np.min(bics), "o", ms=6,
                   mfc="none", mec="#c0392b", mew=1.5)
        inset.set_xlabel("components", fontsize=7)
        inset.set_ylabel("BIC", fontsize=7)
        inset.tick_params(labelsize=6)
    return ax


def plot_gridness_vs_directional(scores_60, resultants, gridness_threshold,
                                 hd_threshold, ax=None):
    """Fig. 1g: every unit as a point, the two cutoffs as dashed lines.

    The top-right quadrant is the conjunctive population; the paper's count
    there is 14, 11% of its 129 grid units.
    """
    ax = ax or plt.gca()
    grid = np.asarray(scores_60) > gridness_threshold
    directional = np.asarray(resultants) > hd_threshold
    both = grid & directional

    ax.scatter(np.asarray(scores_60)[~both], np.asarray(resultants)[~both],
               s=9, alpha=0.45, color="#1a4f7a", linewidths=0)
    ax.scatter(np.asarray(scores_60)[both], np.asarray(resultants)[both],
               s=22, color="#c0392b", linewidths=0,
               label=f"conjunctive: {int(both.sum())}")
    ax.axvline(gridness_threshold, color="k", ls="--", lw=1)
    ax.axhline(hd_threshold, color="k", ls="--", lw=1)
    ax.text(gridness_threshold, ax.get_ylim()[1], f" {gridness_threshold}",
            va="top", fontsize=8)
    ax.text(ax.get_xlim()[1], hd_threshold, f"{hd_threshold} ", ha="right",
            va="bottom", fontsize=8)
    ax.set_xlabel("gridness")
    ax.set_ylabel("length of resultant vector")
    ax.legend(fontsize=8, loc="upper left")
    return ax


def plot_stability(stab, scores_60, resultants, gridness_threshold,
                   hd_threshold, ax=None, bins=25):
    """Ext. Data Fig. 3b: grid-like units hold their map across training,
    directional units do not."""
    ax = ax or plt.gca()
    stab = np.asarray(stab, dtype=float)
    groups = [("all units", np.isfinite(stab), "#999999"),
              ("grid-like", np.asarray(scores_60) > gridness_threshold, "#1a4f7a"),
              ("directional", np.asarray(resultants) > hd_threshold, "#27ae60")]
    edges = np.linspace(-1, 1, bins + 1)
    for label, mask, colour in groups:
        sel = stab[mask & np.isfinite(stab)]
        if len(sel) == 0:
            continue
        ax.hist(sel, bins=edges, alpha=0.55, color=colour,
                label=f"{label}: n={len(sel)}, mean r={sel.mean():.2f}")
    ax.set_xlabel("ratemap correlation, 2e5 vs 3e5 steps")
    ax.set_ylabel("units")
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------
# Path integration
# --------------------------------------------------------------------------

TRAINED_C, UNTRAINED_C, FLOOR_C, TRUE_C = "#1a4f7a", "#c0392b", "#555555", "#7fb3d5"


def plot_path_integration(err, true_pos, decoded, best_mode, decoders,
                          n_examples=4, dt=0.15, paper_final_cm=16):
    """Trajectories, error accumulation, and final-error distribution.

    `err` and `decoded` are both {condition: {decoder: ...}} over trained /
    untrained / floor. Half the example trajectories are drawn from the
    untrained control, so the trained ones have something to be read against.
    """
    n_steps = true_pos.shape[1]
    t = np.arange(n_steps) * dt
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, n_examples, height_ratios=[1, 1.1], hspace=0.35)
    fig.suptitle(f"Path integration (decoder: {best_mode})", fontsize=14)

    # Row 1: example trajectories, true vs decoded -- the paper's Fig. 1b.
    # First half trained, second half the same trajectories untrained.
    n_trained = n_examples // 2
    for i in range(n_examples):
        cond = "trained" if i < n_trained else "untrained"
        traj = i if i < n_trained else i - n_trained
        colour = TRAINED_C if cond == "trained" else UNTRAINED_C
        ax = fig.add_subplot(gs[0, i])
        true, dec = true_pos[traj], decoded[cond][best_mode][traj]
        ax.plot(true[:, 0], true[:, 1], color=TRUE_C, lw=2.5, label="actual")
        ax.plot(dec[:, 0], dec[:, 1], color=colour, lw=1.2, label="decoded")
        ax.plot(*true[0], "o", color=colour, ms=6)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
        ax.set_title(f"{cond} — traj {traj}\nfinal error "
                     f"{100 * np.linalg.norm(dec[-1] - true[-1]):.0f}cm",
                     fontsize=9, color=colour)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Row 2 left: how error accumulates -- the diagnostic a single
    # end-of-trajectory number cannot give.
    ax = fig.add_subplot(gs[1, :2])
    styles = {"trained": ("-", TRAINED_C), "untrained": ("-", UNTRAINED_C),
              "floor": ("--", FLOOR_C)}
    for cond, (ls, c) in styles.items():
        for mode in decoders:
            ax.plot(t, 100 * err[cond][mode].mean(axis=0), ls, color=c,
                    alpha=1.0 if mode == best_mode else 0.28,
                    lw=2.0 if mode == best_mode else 1.0,
                    label=f"{cond} ({mode})" if mode == best_mode else None)
    ax.axhline(paper_final_cm, color="#27ae60", lw=1, ls=":",
               label=f"paper: {paper_final_cm}cm")
    ax.set_xlabel("time (s)"); ax.set_ylabel("mean decoding error (cm)")
    ax.set_title("Error accumulation (bold = best decoder, faint = other two)",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Row 2 right: distribution of end-of-trajectory error -- Fig. 1c.
    ax = fig.add_subplot(gs[1, 2:])
    bins = np.linspace(0, max(200, 100 * err["untrained"][best_mode][:, -1].max()), 60)
    n = err["trained"][best_mode].shape[0]
    ax.hist(100 * err["untrained"][best_mode][:, -1], bins=bins, alpha=0.65,
            color=UNTRAINED_C, label="before training")
    ax.hist(100 * err["trained"][best_mode][:, -1], bins=bins, alpha=0.8,
            color=TRAINED_C, label="after training")
    ax.axvline(100 * err["floor"][best_mode][:, -1].mean(), color=FLOOR_C, ls="--",
               label="decoder floor")
    ax.set_xlabel("error at end of trajectory (cm)")
    ax.set_ylabel(f"trajectories (n={n} per series)")
    ax.set_title("Final-error distribution", fontsize=10)
    ax.legend(fontsize=8)
    return fig
