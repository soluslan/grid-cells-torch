import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from scores import circle_mask

CMAP = "jet"


def sac_plotting_mask(nbins):
    return np.where(circle_mask([nbins * 2 - 1] * 2, nbins), 1.0, np.nan)


def plot_ratemap(ratemap, ax=None, title=None, **kwargs):
    ax = ax or plt.gca()
    ax.imshow(ratemap, interpolation="none", cmap=kwargs.pop("cmap", CMAP), **kwargs)
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=8)
    return ax


def plot_sac(sac, nbins, mask_params=None, ax=None, title=None, **kwargs):
    ax = ax or plt.gca()
    ax.imshow(sac * sac_plotting_mask(nbins), interpolation="none",
              cmap=kwargs.pop("cmap", CMAP), **kwargs)
    if mask_params is not None:
        centre = nbins - 1
        for radius in mask_params:
            ax.add_artist(plt.Circle((centre, centre), radius, fill=False,
                                     edgecolor="k", lw=0.6))
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=8)
    return ax


def plot_hd_tuning(tuning, ax=None, title=None, color="#1a4f7a"):
    ax = ax or plt.gca()
    r = np.asarray(tuning, dtype=float)
    r = r - r.min()
    n = len(r)
    theta = -np.pi + (np.arange(n) + 0.5) * (2 * np.pi / n)
    ax.plot(np.append(theta, theta[0]), np.append(r, r[0]), color=color, lw=1.2)
    ax.fill(np.append(theta, theta[0]), np.append(r, r[0]), color=color, alpha=0.25)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(alpha=0.3, lw=0.4)
    if title is not None:
        ax.set_title(title, fontsize=8)
    return ax


def plot_unit_grid(ratemaps, sacs, mask_params, scores, nbins, indices=None,
                   cols=8, threshold=None, hd_tuning=None, panel_size=1.6,
                   title=None):
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
    fig = plot_unit_grid(**kwargs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_score_distribution(scores_by_layer, threshold, ax=None, bins=60):
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


def plot_scale_distribution(scales, means=None, bics=None, ax=None, bins=20):
    ax = ax or plt.gca()
    s = np.asarray(scales, dtype=float)
    s = s[np.isfinite(s)]
    ax.hist(100 * s, bins=bins, color="#1a4f7a", alpha=0.65, density=True)

    if means is not None and len(means):
        top = ax.get_ylim()[1]
        for m in means:
            ax.axvline(100 * m, color="#c0392b", ls="--", lw=1.2)
            ax.text(100 * m, top * 0.99, f"{100*m:.0f}", ha="center", va="top",
                    fontsize=8, color="#c0392b")
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


TRAINED_C, UNTRAINED_C, FLOOR_C, TRUE_C = "#1a4f7a", "#c0392b", "#555555", "#7fb3d5"


def plot_path_integration(err, true_pos, decoded, best_mode, decoders,
                          n_examples=4, dt=0.15, paper_final_cm=16):
    n_steps = true_pos.shape[1]
    t = np.arange(n_steps) * dt
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, n_examples, height_ratios=[1, 1.1], hspace=0.35)
    fig.suptitle(f"Path integration (decoder: {best_mode})", fontsize=14)

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
