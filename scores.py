"""Per-unit spatial measures: ratemaps, spatial autocorrelograms, gridness.

This file is the paper's "Neuroscience-based analyses of units"
(Supplementary Methods 3d), in the order that section presents them:
gridness, grid scale, head-direction tuning, border score, clustering of
scale, and the inter-trial stability from the main Methods. Measurement
only: no matplotlib, no model, no I/O. Plotting lives in figures.py.

Two of these measures assume a firing rate and so a non-negative intensity,
which a linear bottleneck does not provide; both take a `shift_to_nonnegative`
argument and both document what that costs. Gridness, grid scale and
stability are unaffected -- they rest on Pearson correlations or on peak
positions, which survive any `a -> p*a + q` with p > 0. See README "Known
gaps" 3.

Ported from google-deepmind/grid-cells' scores.py (pure numpy/scipy, no TF),
with modern scipy import paths and two scoring details changed to follow the
PAPER where it disagrees with the released code -- the gridness formula and
the annulus radii. See README "Following the paper, not the released code".
"""

import math

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats

# Paper's expanding annulus, in ratemap bins (Supplementary Methods 3d):
# "radius of 8 bins and with the central peak excluded [...] expanding the
# annulus by 2, up to a maximum of 20."
PAPER_OUTER_RADII_BINS = tuple(range(8, 21, 2))

# The paper excludes "the central peak" without giving its radius; this is a
# choice made here, ported from the released code's fractional 0.2 (at its own
# nbins=20 -- not rescaled to ours). Consequential, not a negligible knob: see
# README "Everything about the representation" for the sensitivity sweep and
# why the default stays put rather than being tuned to match the paper.
DEFAULT_INNER_RADIUS_BINS = 4.0


def paper_mask_parameters(inner_radius_bins=DEFAULT_INNER_RADIUS_BINS,
                          outer_radii_bins=PAPER_OUTER_RADII_BINS):
    """The paper's mask set as (inner, outer) radii in absolute ratemap bins."""
    return [(float(inner_radius_bins), float(r)) for r in outer_radii_bins]


def circle_mask(size, radius):
    """Boolean mask of a circle of `radius` bins, centered in a `size` grid."""
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.expand_dims(np.linspace(-sz[0], sz[1], size[1]), 0).repeat(size[0], 0)
    y = np.expand_dims(np.linspace(-sz[0], sz[1], size[1]), 1).repeat(size[1], 1)
    return np.sqrt(x ** 2 + y ** 2) <= radius


class GridScorer(object):
    """Scores ratemaps for hexagonal (grid-cell-like) periodicity."""

    def __init__(self, nbins, coords_range, mask_parameters):
        """
        Args:
            nbins: ratemap bins per dimension (the paper's Methods uses 32).
            coords_range: ((xmin,xmax),(ymin,ymax)) environment extent.
            mask_parameters: (inner, outer) annulus radii in absolute bins --
                see `paper_mask_parameters()`. The best-scoring ring wins.
        """
        self.nbins = nbins
        self._coords_range = coords_range
        self._corr_angles = [30, 60, 90, 120, 150]
        self._masks = [(self._get_ring_mask(lo, hi), (lo, hi)) for lo, hi in mask_parameters]

    def calculate_ratemap(self, xs, ys, activations, statistic="mean"):
        return scipy.stats.binned_statistic_2d(
            xs, ys, activations, bins=self.nbins, statistic=statistic,
            range=self._coords_range)[0]

    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self.nbins * 2 - 1] * 2
        return (circle_mask(n_points, mask_max) & ~circle_mask(n_points, mask_min)).astype(float)

    def grid_score_60(self, corr):
        """Paper's formula (Supplementary Methods 3d): "the highest correlation
        obtained from rotations of 30, 90 and 150 deg subtracted from the
        lowest at 0, 60 and 120 deg". The 0 deg term is omitted: it is ~1.0 by
        construction and every other correlation is <=1, so it never binds.
        """
        return np.minimum(corr[60], corr[120]) - np.maximum(
            corr[30], np.maximum(corr[90], corr[150]))

    def calculate_sac(self, rate_map):
        """Spatial autocorrelogram of a ratemap: sliding Pearson correlation of
        the map against itself at every offset.

        The original wrote this as a general two-sequence cross-correlation and
        then passed the same map twice, plus masks meant to exclude NaN bins.
        Those masks are dead there and here: `nan_to_num` runs first, so the
        `isnan` tests that build them never fire. Unvisited bins are therefore
        folded to 0 rather than excluded -- inherited behaviour, kept, and of
        little consequence at 4000 trajectories (~390 samples per bin, so empty
        bins are rare). Verified numerically identical to the original form.
        """
        def filter2(b, x):
            return scipy.signal.convolve2d(x, np.rot90(b, 2), mode="full")

        seq = np.nan_to_num(rate_map)
        ones = np.ones(seq.shape)
        seq_sq = np.square(seq)

        # filter2 is not symmetric in its arguments, so the two "sum" and two
        # "sum of squares" terms differ even though the sequences are equal.
        seq1_x_seq2 = filter2(seq, seq)
        sum_seq1 = filter2(seq, ones)
        sum_seq2 = filter2(ones, seq)
        sum_seq1_sq = filter2(seq_sq, ones)
        sum_seq2_sq = filter2(ones, seq_sq)
        n_bins = filter2(ones, ones)
        n_bins_sq = np.square(n_bins)

        # Bins with ~0 overlap or ~0 local variance give 0/0 and x/0 here;
        # expected, and cleaned up by the nan_to_num below.
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
        # posinf/neginf=0, not nan_to_num's huge-finite default: an undefined
        # correlation means "no information" (0), and a 1.8e308 would be
        # spline-interpolated by scipy.ndimage.rotate across a wide area.
        return np.nan_to_num(np.real(x_coef), posinf=0.0, neginf=0.0)

    def rotated_sacs(self, sac, angles):
        return [scipy.ndimage.rotate(sac, angle, reshape=False) for angle in angles]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        masked_sac_mean = np.sum(masked_sac) / ring_area
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered ** 2) / ring_area + 1e-5
        corrs = {}
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            corrs[angle] = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area / variance
        return self.grid_score_60(corrs)

    def get_scores(self, rate_map):
        """-> (score_60, best_mask_params, sac); the best-scoring ring wins."""
        sac = self.calculate_sac(rate_map)
        rotated = self.rotated_sacs(sac, self._corr_angles)
        scores = np.asarray([self.get_grid_scores_for_mask(sac, rotated, mask)
                             for mask, _ in self._masks])
        best = int(np.argmax(scores))
        return scores[best], self._masks[best][1], sac


HD_BINS = 20  # paper's Methods: "directional bins as 20 equal width intervals"
HD_THRESHOLD = 0.47  # Suppl. 3d: Rayleigh test of directional uniformity, alpha=0.01


def directional_ratemap(headings: np.ndarray, activations: np.ndarray,
                        n_bins: int = HD_BINS) -> np.ndarray:
    """Mean activation per heading bin -- the paper's "directional activity map".

    headings [N] in radians, activations [N, n_units] -> [n_units, n_bins].
    The spatial ratemap's one-dimensional twin: same averaging, binned by which
    way the agent faced instead of where it stood.
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(headings, edges) - 1, 0, n_bins - 1)
    onehot = np.zeros((len(idx), n_bins), dtype=np.float32)
    onehot[np.arange(len(idx)), idx] = 1.0
    counts = onehot.sum(axis=0)
    sums = onehot.T @ activations  # [n_bins, n_units]
    return (sums / np.maximum(counts, 1)[:, None]).T


def resultant_vector_length(tuning: np.ndarray, shift_to_nonnegative: bool = True
                            ) -> np.ndarray:
    """How directionally tuned a unit is: Suppl. Methods 3d eqs. (6)-(7).

    Each bin contributes a vector of length `intensity` pointing at the bin's
    angle; the resultant is their sum divided by the total intensity, so a unit
    firing equally in all directions scores 0 and one firing in a single
    direction scores 1. The paper calls anything above 0.47 directionally tuned.

    `tuning` is [.., n_bins]; returns [..].

    NOT TRUSTWORTHY HERE. The formula assumes a non-negative firing rate; our
    bottleneck is a plain linear layer with signed activations, so
    `shift_to_nonnegative` subtracts each unit's own minimum first -- our
    choice, the paper never says how it handled this. But for a cosine-shaped
    tuning curve `A*cos(a-phi) + B`, that specific shift collapses the result
    to exactly 1/2 regardless of A or B: the shift deletes the baseline the
    ratio is supposed to measure modulation against. Since 0.5 > the paper's
    0.47 cutoff, any faintly-tuned unit reads as directionally tuned. See
    README "Known gaps" 3 for the derivation and the measurements.
    """
    tuning = np.asarray(tuning, dtype=float)
    if shift_to_nonnegative:
        tuning = tuning - tuning.min(axis=-1, keepdims=True)
    n = tuning.shape[-1]
    angles = -np.pi + (np.arange(n) + 0.5) * (2 * np.pi / n)  # bin centres
    total = tuning.sum(axis=-1)
    x = (tuning * np.cos(angles)).sum(axis=-1)
    y = (tuning * np.sin(angles)).sum(axis=-1)
    # A perfectly flat unit shifts to all zeros: no preferred direction, so 0.
    return np.where(total == 0, 0.0,
                    np.sqrt(x ** 2 + y ** 2) / np.where(total == 0, 1.0, total))


# --------------------------------------------------------------------------
# Grid scale -- Suppl. Methods 3d
# --------------------------------------------------------------------------

def sac_local_maxima(sac: np.ndarray, within_radius: float | None = None) -> np.ndarray:
    """Strict local maxima of a SAC as [k, 2] (row, col) integer coordinates.

    Strict (`>` every 8-neighbour, centre excluded from the comparison) rather
    than `>=`, so the zero plateaus that `calculate_sac` leaves in the corners
    contribute nothing. `within_radius` drops maxima outside the region of the
    autocorrelogram that the ratemap ever populates.
    """
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False  # compare against neighbours only, not self
    neighbour_max = scipy.ndimage.maximum_filter(sac, footprint=footprint,
                                                 mode="constant", cval=-np.inf)
    peaks = np.argwhere(sac > neighbour_max)
    if within_radius is not None and len(peaks):
        centre = (np.asarray(sac.shape) - 1) / 2.0
        keep = np.linalg.norm(peaks - centre, axis=1) <= within_radius
        peaks = peaks[keep]
    return peaks


def grid_scale(sac: np.ndarray, bin_size_m: float, n_peaks: int = 6,
               search_radius_bins: float = PAPER_OUTER_RADII_BINS[-1]) -> float:
    """Wavelength of the spatial periodicity, in METRES (Suppl. Methods 3d):
    "The six local maxima closest to but excluding the central peak were
    identified. Grid scale was then calculated as the median distance of these
    peaks from the origin."

    Returns NaN when fewer than `n_peaks` off-centre maxima exist -- a unit
    with no periodic structure has no scale, and a median over 2 or 3 stray
    maxima would be a number without a meaning.

    `search_radius_bins` defaults to 20, the paper's widest gridness annulus.
    Beyond it the SAC is unreliable: far offsets overlap the ratemap in a thin
    strip that can correlate almost perfectly by construction, not because of
    real periodicity (an earlier bug: a non-periodic blob was assigned a
    spurious 204cm "scale" before this cap existed). Feed this only units that
    already pass the gridness cutoff, as the paper's Reporting Summary
    specifies -- the cap prevents nonsense on a non-grid unit, but doesn't make
    its scale meaningful.

    `bin_size_m` is the ratemap's bin width (env_size / nbins); SAC pixels and
    ratemap bins are the same unit, so the conversion is a single multiply.
    """
    centre = (np.asarray(sac.shape) - 1) / 2.0
    peaks = sac_local_maxima(sac, within_radius=search_radius_bins)
    if len(peaks) == 0:
        return float("nan")

    distances = np.linalg.norm(peaks - centre, axis=1)
    # Drop the central peak itself. It sits exactly at the centre, so anything
    # at distance 0 is it; a strict > 0 test needs no tolerance parameter.
    distances = distances[distances > 0]
    if len(distances) < n_peaks:
        return float("nan")
    return float(np.median(np.sort(distances)[:n_peaks]) * bin_size_m)


# --------------------------------------------------------------------------
# Border score -- Suppl. Methods 3d eq. (8)
# --------------------------------------------------------------------------

BORDER_THRESHOLD = 0.50  # Suppl. 3d, same field-shuffle derivation as 0.37
BORDER_BINS = 20  # "In all our experiments 20 by 20 bins where used"
BORDER_WALL_BINS = 3  # "and d_b took value 3"


def border_score(ratemap: np.ndarray, d_b: int = BORDER_WALL_BINS,
                 shift_to_nonnegative: bool = True) -> float:
    """How much more active a unit is against one wall than in the middle:

        b_s = max_i (b_i - c) / (b_i + c)

    over the four walls, where b_i averages the bins within `d_b` of wall i and
    c averages the bins further than `d_b` from every wall.

    NOTE `shift_to_nonnegative`. Unlike gridness, this ratio assumes b_i, c are
    non-negative firing rates: with signed activations the denominator can
    cross zero, and when both terms are negative the score's SIGN inverts
    (b=-1, c=-3 scores -0.5 for a unit that is MORE active at the wall).
    Subtracting the unit's own ratemap minimum restores b_i, c >= 0 -- our
    choice, like the head-direction measure, since the paper never says how it
    handled this.

    The paper scores borders on 20x20 bins, not the 32x32 the same section
    uses for gridness, so pass a ratemap built at `BORDER_BINS`.
    """
    rm = np.asarray(ratemap, dtype=float)
    if shift_to_nonnegative:
        rm = rm - np.nanmin(rm)

    n = rm.shape[0]
    assert d_b * 2 < n, f"d_b={d_b} leaves no interior in a {n}x{n} ratemap"
    walls = (rm[:d_b, :], rm[-d_b:, :], rm[:, :d_b], rm[:, -d_b:])
    c = np.nanmean(rm[d_b:n - d_b, d_b:n - d_b])

    scores = []
    for wall in walls:
        b = np.nanmean(wall)
        denom = b + c
        scores.append(0.0 if denom == 0 else (b - c) / denom)
    return float(np.max(scores))


# --------------------------------------------------------------------------
# Clustering of scale -- Suppl. Methods 3d, Fig. 1e
# --------------------------------------------------------------------------

DISCRETENESS_BINS = 13  # "a histogram with 13 bins"


def discreteness(scales: np.ndarray, n_bins: int = DISCRETENESS_BINS,
                 value_range: tuple[float, float] | None = None) -> float:
    """Suppl. 3d: "'Discreteness' was defined as the standard deviation of the
    counts in each bin". High when scales pile into a few bins, low when they
    spread evenly -- so it separates a clustered population from a continuous
    one without assuming how many clusters there are.

    `value_range` defaults to the data's own extent. The paper instead fixes
    it at "scales 10 to 36 spatial bins", which doesn't reconcile with its own
    reported 28-115cm range at any consistent bin width (see README for the
    arithmetic). Working in metres and letting the range follow the data
    avoids inheriting that contradiction; pass an explicit range to compare
    two populations on identical bins.
    """
    scales = np.asarray(scales, dtype=float)
    scales = scales[np.isfinite(scales)]
    if len(scales) == 0:
        return float("nan")
    counts, _ = np.histogram(scales, bins=n_bins, range=value_range)
    return float(np.std(counts))


def discreteness_significance(scales: np.ndarray, n_shuffles: int = 500,
                              n_bins: int = DISCRETENESS_BINS, seed: int = 0
                              ) -> tuple[float, np.ndarray, float]:
    """-> (observed discreteness, null distribution [n_shuffles], p).

    Suppl. 3d's null: "a random number was drawn from a flat distribution
    between -1/2 and +1/2 of the smallest grid scale [...] added to the grid
    scales, the population was binned as before, and the discreteness score
    calculated." This smears real clustering while preserving the overall
    spread, so a high score survives only if the peaks are real. p is the
    fraction of shuffles reaching the observed value ("exceeded that of all
    the 500 shuffles (p < 0.002)" in the paper, read as one-sided).

    Every histogram spans its OWN data's extent, observed and shuffled alike --
    sharing one range instead leaves the observed scales' outer bins empty
    while the jitter fills them in, inflating the observed SD by edge effects
    alone (an earlier bug this fixes).
    """
    scales = np.asarray(scales, dtype=float)
    scales = scales[np.isfinite(scales)]
    if len(scales) == 0:
        return float("nan"), np.array([]), float("nan")

    half = scales.min() / 2.0
    observed = discreteness(scales, n_bins=n_bins)

    rng = np.random.default_rng(seed)
    null = np.array([
        discreteness(scales + rng.uniform(-half, half, size=len(scales)),
                     n_bins=n_bins)
        for _ in range(n_shuffles)])
    p = float((null >= observed).sum() + 1) / (n_shuffles + 1)
    return observed, null, p


def cluster_scales(scales: np.ndarray, max_components: int = 8, seed: int = 0):
    """Fit 1..max_components Gaussians and keep the lowest-BIC model.

    Suppl. 3d: "the distribution of scales from grid-like units was fit with
    Gaussian mixture distributions containing 1 to 8 components [...] compared
    using Bayesian Information Criterion, the model (3 components) with the
    lowest BIC score was selected as the most efficient."

    -> (n_components, sorted cluster means, bic per component count [1..max],
        ratios between neighbouring means). The paper's answer: 3 clusters at
        0.47/0.70/1.06m with neighbour ratios near 1.5.
    """
    from sklearn.mixture import GaussianMixture

    scales = np.asarray(scales, dtype=float)
    scales = scales[np.isfinite(scales)].reshape(-1, 1)
    if len(scales) < max_components:
        return 0, np.array([]), np.array([]), np.array([])

    bics, models = [], []
    for k in range(1, max_components + 1):
        gm = GaussianMixture(n_components=k, random_state=seed, n_init=5).fit(scales)
        bics.append(gm.bic(scales))
        models.append(gm)

    best = int(np.argmin(bics))
    means = np.sort(models[best].means_.ravel())
    ratios = means[1:] / means[:-1] if len(means) > 1 else np.array([])
    return best + 1, means, np.array(bics), ratios


# --------------------------------------------------------------------------
# Inter-trial stability -- Methods, Ext. Data Fig. 3b
# --------------------------------------------------------------------------

def ratemap_stability(ratemap_a: np.ndarray, ratemap_b: np.ndarray) -> float:
    """Pearson correlation between two ratemaps of the same unit, bin by bin.

    Methods: "the reliability of spatial firing between baseline trials was
    assessed by calculating the spatial correlation between pairs of rate maps
    taken at two different logging steps in training (t = 2 x 10^5,
    t' = 3 x 10^5) [...] The Pearson product moment correlation coefficient was
    calculated between equivalent bins in the two trials and unvisited bins
    were excluded from the measure."

    Unvisited bins arrive as NaN from `binned_statistic_2d` and are dropped
    here rather than folded to 0 -- unlike inside `calculate_sac`, where the
    inherited behaviour is to zero them. Returns NaN if either map is constant
    over the shared bins, where the correlation is undefined.
    """
    a = np.asarray(ratemap_a, dtype=float).ravel()
    b = np.asarray(ratemap_b, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    a, b = a[valid], b[valid]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_units(scorer: GridScorer, xy: np.ndarray, activations: np.ndarray):
    """Score every unit of a layer.

    xy [N,2] and activations [N, n_units] are one row per (trajectory,
    timestep). Returns (scores_60 [n_units], ratemaps, sacs, mask_params) --
    the intermediates come back because every one of them gets plotted.
    """
    ratemaps = [scorer.calculate_ratemap(xy[:, 0], xy[:, 1], activations[:, i])
                for i in range(activations.shape[1])]
    results = [scorer.get_scores(rm) for rm in ratemaps]
    return (np.array([r[0] for r in results]), ratemaps,
            [r[2] for r in results], [r[1] for r in results])
