"""Grid score calculations: ratemaps, spatial autocorrelograms, gridness.

Ported from google-deepmind/grid-cells' scores.py. This file has no
TensorFlow/Sonnet dependency in the original -- it's pure numpy/scipy/
matplotlib -- so the only changes needed are modern scipy import paths:
  - `scipy.ndimage.interpolation.rotate` -> `scipy.ndimage.rotate` (the
    `interpolation` submodule alias is deprecated/gone in current scipy)
  - explicit `import scipy.stats` / `import scipy.ndimage` (the original
    relied on these being transitively importable via `import scipy.signal`,
    which current scipy no longer guarantees)
Otherwise this is a 1:1 port; it operates on plain numpy arrays throughout,
so it works unchanged on ratemaps computed from PyTorch activations (just
call `.detach().cpu().numpy()` first).
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
        return np.nan_to_num(x_coef)

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
