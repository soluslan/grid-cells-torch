"""Grid score calculations: ratemaps, spatial autocorrelograms, gridness.

Ported from google-deepmind/grid-cells' scores.py. This file has no
TensorFlow/Sonnet dependency in the original -- it's pure numpy/scipy/
matplotlib -- so the only changes needed are modern scipy import paths:
  - `scipy.ndimage.interpolation.rotate` -> `scipy.ndimage.rotate` (the
    `interpolation` submodule alias is deprecated/gone in current scipy)
  - explicit `import scipy.stats` / `import scipy.ndimage` (the original
    relied on these being transitively importable via `import scipy.signal`,
    which current scipy no longer guarantees)
  - `calculate_sac`'s final `nan_to_num` now fills +-inf with 0 instead of
    numpy's default +-1.8e308: a bin's correlation is only ever ill-defined
    (0/0 or x/0) where local overlap variance is ~0, and "no information"
    (0) is the right value there, not a near-float-max number that
    `scipy.ndimage.rotate` would then spline-interpolate into NaN over a
    much wider area than the single offending bin. This surfaced while
    scoring field-shuffled ratemaps below (large flat backgrounds make
    near-zero-variance edge bins common) but is a latent correctness issue
    for any sparse/patchy ratemap, not just shuffled ones.
Otherwise this is a 1:1 port; it operates on plain numpy arrays throughout,
so it works unchanged on ratemaps computed from PyTorch activations (just
call `.detach().cpu().numpy()` first).

`field_labels`/`shuffle_fields`/`GridScorer.shuffled_gridness_threshold`
below are NOT part of the original port -- google-deepmind/grid-cells never
released this code. They reimplement the paper's own significance test
(Banino et al. 2018 Supplementary Methods 3d): per unit, watershed-segment
its ratemap into firing fields, relocate each field's peak to a random bin
100 times, and take the 95th percentile of the resulting gridness
distribution as that unit's own threshold -- instead of one fixed cutoff
(e.g. 0.37) applied uniformly to every unit. The watershed step here uses a
steepest-ascent walk (every bin flows to the local peak reached by always
stepping to the highest neighbour) rather than `scipy.ndimage.watershed_ift`
or skimage's `watershed`, to avoid adding a dependency -- it produces the
same drainage-basin partition.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
    """Boolean-ish mask of a circle of the given radius centered in `size`."""
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0).repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1).repeat(size[1], 1)
    z = np.sqrt(x ** 2 + y ** 2)
    z = np.less_equal(z, radius)
    return np.where(z, in_val, out_val)


def _steepest_ascent_basins(filled):
    """Partitions a 2D array into drainage basins, one per local maximum.

    Every bin is assigned the label of the local peak reached by repeatedly
    stepping to the highest of its 8 neighbours (ties broken by scan order).
    This is a dependency-free equivalent of watershed-by-flooding. Returns
    (labels, n_basins).
    """
    ny, nx = filled.shape
    label = np.full((ny, nx), -1, dtype=int)
    next_label = 0
    for i in range(ny):
        for j in range(nx):
            if label[i, j] != -1:
                continue
            path = []
            ci, cj = i, j
            while True:
                if label[ci, cj] != -1:
                    result = label[ci, cj]
                    break
                path.append((ci, cj))
                best_i, best_j, best_val = ci, cj, filled[ci, cj]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < ny and 0 <= nj < nx and filled[ni, nj] > best_val:
                            best_i, best_j, best_val = ni, nj, filled[ni, nj]
                if (best_i, best_j) == (ci, cj):
                    result = next_label
                    next_label += 1
                    break
                ci, cj = best_i, best_j
            for (pi, pj) in path:
                label[pi, pj] = result
    return label, next_label


def field_labels(ratemap, min_peak_frac=0.2):
    """Segments a ratemap into firing fields.

    Each field is the drainage basin of one local peak (see
    `_steepest_ascent_basins`), restricted to the bins at or above
    `min_peak_frac` of that peak's height -- the standard "field = X% of
    peak" boundary convention. Singleton (1-bin) fields are dropped as
    noise. Returns (labels, n_fields); label 0 is background (everything
    outside any field).
    """
    filled = np.nan_to_num(ratemap, nan=0.0)
    basins, n_basins = _steepest_ascent_basins(filled)
    labels = np.zeros_like(basins)
    next_label = 1
    for b in range(n_basins):
        mask = basins == b
        peak_val = filled[mask].max()
        if peak_val <= 0:
            continue
        field_mask = mask & (filled >= min_peak_frac * peak_val)
        if field_mask.sum() >= 2:
            labels[field_mask] = next_label
            next_label += 1
    return labels, next_label - 1


def shuffle_fields(ratemap, labels, n_fields, rng):
    """One draw from the per-unit field-shuffle null distribution: relocate
    every field's peak to a uniformly random bin (keeping its shape, clipped
    at the map edges), and refill everywhere else -- including the vacated
    original field footprints -- with the ratemap's own background
    (non-field) mean level, so the shuffled map preserves the original's
    overall topology/statistics outside the relocated fields. Overlapping
    relocated fields combine via max (like overlapping firing bumps).
    """
    filled = np.nan_to_num(ratemap, nan=0.0)
    ny, nx = filled.shape
    background_mask = labels == 0
    background_val = filled[background_mask].mean() if background_mask.any() else 0.0
    shuffled = np.full((ny, nx), background_val, dtype=filled.dtype)

    for f in range(1, n_fields + 1):
        ys, xs = np.where(labels == f)
        peak_idx = np.argmax(filled[ys, xs])
        peak_y, peak_x = ys[peak_idx], xs[peak_idx]
        rel_y, rel_x, vals = ys - peak_y, xs - peak_x, filled[ys, xs]

        new_y, new_x = rng.integers(0, ny), rng.integers(0, nx)
        ty, tx = new_y + rel_y, new_x + rel_x
        valid = (ty >= 0) & (ty < ny) & (tx >= 0) & (tx < nx)
        shuffled[ty[valid], tx[valid]] = np.maximum(shuffled[ty[valid], tx[valid]], vals[valid])

    shuffled[np.isnan(ratemap)] = np.nan
    return shuffled


class GridScorer(object):
    """Scores ratemaps for hexagonal (grid-cell-like) periodicity."""

    def __init__(self, nbins, coords_range, mask_parameters, min_max=False):
        """
        Args:
            nbins: number of bins per dimension in the ratemap.
            coords_range: ((xmin,xmax),(ymin,ymax)) environment extent.
            mask_parameters: iterable of (mask_min, mask_max) ring radii
                (as a fraction of nbins) to try; the best-scoring ring wins.
            min_max: use the original repo's alternate scoring formula.
        """
        self._nbins = nbins
        self._min_max = min_max
        self._coords_range = coords_range
        self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
        self._masks = [(self._get_ring_mask(mask_min, mask_max), (mask_min, mask_max))
                       for mask_min, mask_max in mask_parameters]
        self._plotting_sac_mask = circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1], self._nbins,
            in_val=1.0, out_val=np.nan)

    def calculate_ratemap(self, xs, ys, activations, statistic="mean"):
        return scipy.stats.binned_statistic_2d(
            xs, ys, activations, bins=self._nbins, statistic=statistic,
            range=self._coords_range)[0]

    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return (circle_mask(n_points, mask_max * self._nbins)
                * (1 - circle_mask(n_points, mask_min * self._nbins)))

    def grid_score_60(self, corr):
        if self._min_max:
            return np.minimum(corr[60], corr[120]) - np.maximum(
                corr[30], np.maximum(corr[90], corr[150]))
        return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

    def grid_score_90(self, corr):
        return corr[90] - (corr[45] + corr[135]) / 2

    def calculate_sac(self, seq1):
        """Spatial autocorrelogram of a ratemap (may contain NaN bins)."""
        seq2 = seq1

        def filter2(b, x):
            stencil = np.rot90(b, 2)
            return scipy.signal.convolve2d(x, stencil, mode="full")

        seq1 = np.nan_to_num(seq1)
        seq2 = np.nan_to_num(seq2)

        ones_seq1 = np.ones(seq1.shape)
        ones_seq1[np.isnan(seq1)] = 0
        ones_seq2 = np.ones(seq2.shape)
        ones_seq2[np.isnan(seq2)] = 0

        seq1[np.isnan(seq1)] = 0
        seq2[np.isnan(seq2)] = 0

        seq1_sq = np.square(seq1)
        seq2_sq = np.square(seq2)

        seq1_x_seq2 = filter2(seq1, seq2)
        sum_seq1 = filter2(seq1, ones_seq2)
        sum_seq2 = filter2(ones_seq1, seq2)
        sum_seq1_sq = filter2(seq1_sq, ones_seq2)
        sum_seq2_sq = filter2(ones_seq1, seq2_sq)
        n_bins = filter2(ones_seq1, ones_seq2)
        n_bins_sq = np.square(n_bins)

        # bins with ~0 overlap (n_bins small) or ~0 local variance produce
        # 0/0 and x/0 here -- expected and harmless, cleaned up by the
        # posinf/neginf=0 nan_to_num below, so silence the routine warnings.
        with np.errstate(invalid="ignore", divide="ignore"):
            std_seq1 = np.power(
                np.subtract(np.divide(sum_seq1_sq, n_bins),
                            np.divide(np.square(sum_seq1), n_bins_sq)), 0.5)
            std_seq2 = np.power(
                np.subtract(np.divide(sum_seq2_sq, n_bins),
                            np.divide(np.square(sum_seq2), n_bins_sq)), 0.5)
            covar = np.subtract(
                np.divide(seq1_x_seq2, n_bins),
                np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
            x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
        x_coef = np.real(x_coef)
        # posinf/neginf=0 (not nan_to_num's huge-finite-number default): a
        # correlation is only ever ill-defined (0/0 or x/0) at bins with
        # near-zero overlap variance, where "no correlation information" is
        # the correct value, not +-1.8e308 -- which downstream
        # scipy.ndimage.rotate would interpolate into NaN over a much wider
        # area than the single offending bin.
        return np.nan_to_num(x_coef, posinf=0.0, neginf=0.0)

    def rotated_sacs(self, sac, angles):
        return [scipy.ndimage.rotate(sac, angle, reshape=False) for angle in angles]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        masked_sac_mean = np.sum(masked_sac) / ring_area
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered ** 2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

    def get_scores(self, rate_map):
        """Returns (score_60, score_90, best_60_mask_params, best_90_mask_params, sac)."""
        sac = self.calculate_sac(rate_map)
        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

        scores = [self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
                  for mask, _ in self._masks]
        scores_60, scores_90, variances = map(np.asarray, zip(*scores))
        max_60_ind = np.argmax(scores_60)
        max_90_ind = np.argmax(scores_90)

        return (scores_60[max_60_ind], scores_90[max_90_ind],
                self._masks[max_60_ind][1], self._masks[max_90_ind][1], sac)

    def shuffled_gridness_threshold(self, ratemap, n_shuffles=100, percentile=95,
                                     min_peak_frac=0.2, rng=None):
        """This unit's own gridness significance threshold (Banino et al.
        2018 Supplementary Methods 3d): segment `ratemap` into fields once,
        then relocate them to random bins `n_shuffles` times and return the
        `percentile`-th percentile of the resulting score_60 distribution.
        A unit is "grid-like" if its real score_60 exceeds this -- its own,
        rather than one fixed cutoff shared by every unit. Returns NaN if
        the ratemap has no detectable fields (nothing to shuffle).
        """
        if rng is None:
            rng = np.random.default_rng()
        labels, n_fields = field_labels(ratemap, min_peak_frac=min_peak_frac)
        if n_fields == 0:
            return np.nan
        null_scores = np.empty(n_shuffles)
        for k in range(n_shuffles):
            shuffled = shuffle_fields(ratemap, labels, n_fields, rng)
            null_scores[k] = self.get_scores(shuffled)[0]
        # a shuffled placement can occasionally starve calculate_sac's local
        # std of overlap (std=0 -> 0/0 -> nan_to_num's +-inf fill-in ->
        # overflow downstream); nanpercentile drops those instead of letting
        # one degenerate draw poison the whole threshold.
        null_scores[~np.isfinite(null_scores)] = np.nan
        if np.all(np.isnan(null_scores)):
            return np.nan
        return np.nanpercentile(null_scores, percentile)

    def plot_ratemap(self, ratemap, ax=None, title=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(ratemap, interpolation="none", *args, **kwargs)
        ax.axis("off")
        if title is not None:
            ax.set_title(title)

    def plot_sac(self, sac, mask_params=None, ax=None, title=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        useful_sac = sac * self._plotting_sac_mask
        ax.imshow(useful_sac, interpolation="none", *args, **kwargs)
        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(plt.Circle((center, center), mask_params[0] * self._nbins,
                                      fill=False, edgecolor="k"))
            ax.add_artist(plt.Circle((center, center), mask_params[1] * self._nbins,
                                      fill=False, edgecolor="k"))
        ax.axis("off")
        if title is not None:
            ax.set_title(title)
